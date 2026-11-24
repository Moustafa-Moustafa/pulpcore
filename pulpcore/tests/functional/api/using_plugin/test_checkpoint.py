"""Tests for checkpoint distribution and publications."""

import hashlib
import re
import uuid
from datetime import datetime, timedelta, timezone
from time import sleep
from urllib.parse import urlparse

import pytest
import requests
from aiohttp import ClientResponseError

from pulpcore.client.pulp_file import exceptions
from pulpcore.content.handler import Handler


def _wait_until_checkpoint_ts_advances(previous_created):
    """Block until wall-clock formats to a later checkpoint timestamp than previous_created.

    Checkpoint URLs are second-resolution, so consecutive publications need distinct
    seconds. Fixed sleep(1) is wasteful when create/publish already crossed a second.
    """
    previous_ts = Handler._format_checkpoint_timestamp(previous_created)
    while Handler._format_checkpoint_timestamp(datetime.now(timezone.utc)) == previous_ts:
        sleep(0.05)


@pytest.fixture(scope="class")
def content_factory(tmp_path_factory, file_bindings):
    def _content_factory(name):
        file = tmp_path_factory.mktemp("content") / name
        file.write_text(str(uuid.uuid4()))
        return file_bindings.ContentFilesApi.upload(relative_path=name, file=str(file)).pulp_href

    def _precreate(names):
        return [_content_factory(name) for name in names]

    _content_factory.precreate = _precreate
    return _content_factory


@pytest.fixture(scope="class")
def create_publication(content_factory, file_bindings, monitor_task):
    counter = [0]
    content_queue = []

    def precreate(n):
        names = []
        for _ in range(n):
            names.append(str(counter[0]))
            counter[0] += 1
        content_queue.extend(content_factory.precreate(names))

    def _create_publication(repo, checkpoint):
        if not content_queue:
            precreate(1)
        content_href = content_queue.pop(0)

        monitor_task(
            file_bindings.RepositoriesFileApi.modify(
                repo.pulp_href, {"add_content_units": [content_href]}
            ).task
        )

        response = monitor_task(
            file_bindings.PublicationsFileApi.create(
                {"repository": repo.pulp_href, "checkpoint": checkpoint}
            ).task
        )
        return file_bindings.PublicationsFileApi.read(response.created_resources[0])

    _create_publication.precreate = precreate
    return _create_publication


@pytest.fixture(scope="class")
def setup(
    file_repository_factory,
    file_distribution_factory,
    create_publication,
):
    repo = file_repository_factory()
    distribution = file_distribution_factory(repository=repo.pulp_href, checkpoint=True)

    # Five publications: content creates overlap; only wait between pubs when needed
    # for distinct second-resolution checkpoint timestamps.
    create_publication.precreate(5)
    pubs = []
    for checkpoint in (False, True, False, True, False):
        if pubs:
            _wait_until_checkpoint_ts_advances(pubs[-1].pulp_created)
        pubs.append(create_publication(repo, checkpoint))

    return pubs, distribution


@pytest.fixture
def checkpoint_url(distribution_base_url):
    def _checkpoint_url(distribution, timestamp):
        distro_base_url = distribution_base_url(distribution.base_url)
        return f"{distro_base_url}{Handler._format_checkpoint_timestamp(timestamp)}/"

    return _checkpoint_url


class TestCheckpointDistribution:
    """Don't mark parallel, tests are shorter than setup."""

    def test_base_path_lists_checkpoints(self, setup, http_get, distribution_base_url):
        pubs, distribution = setup

        response = http_get(distribution_base_url(distribution.base_url)).decode("utf-8")

        checkpoints_ts = set(re.findall(r"\d{8}T\d{6}Z", response))
        assert len(checkpoints_ts) == 2
        assert Handler._format_checkpoint_timestamp(pubs[1].pulp_created) in checkpoints_ts
        assert Handler._format_checkpoint_timestamp(pubs[3].pulp_created) in checkpoints_ts

    def test_distro_root_no_trailing_slash_is_redirected(
        self,
        setup,
        http_get,
        distribution_base_url,
    ):
        """Test checkpoint listing when path doesn't end with a slash."""

        pubs, distribution = setup

        # Test a checkpoint distro listing path
        response = http_get(distribution_base_url(distribution.base_url[:-1])).decode("utf-8")
        checkpoints_ts = set(re.findall(r"\d{8}T\d{6}Z", response))

        assert len(checkpoints_ts) == 2
        assert Handler._format_checkpoint_timestamp(pubs[1].pulp_created) in checkpoints_ts
        assert Handler._format_checkpoint_timestamp(pubs[3].pulp_created) in checkpoints_ts

    def test_timestamped_checkpoint_no_trailing_slash_is_redirected(
        self,
        setup,
        http_get,
        checkpoint_url,
    ):
        """Test a timestamped checkpoint when path doesn't end with a slash."""

        pubs, distribution = setup

        pub_1_url = checkpoint_url(distribution, pubs[1].pulp_created)
        response = http_get(pub_1_url[:-1]).decode("utf-8")

        assert f"<h1>Index of {urlparse(pub_1_url).path}</h1>" in response

    def test_exact_timestamp_is_served(self, setup, http_get, checkpoint_url):
        pubs, distribution = setup

        pub_1_url = checkpoint_url(distribution, pubs[1].pulp_created)
        response = http_get(pub_1_url).decode("utf-8")

        assert f"<h1>Index of {urlparse(pub_1_url).path}</h1>" in response

    def test_invalid_timestamp_returns_404(self, setup, http_get, distribution_base_url):
        _, distribution = setup
        with pytest.raises(ClientResponseError) as exc:
            http_get(distribution_base_url(f"{distribution.base_url}invalid_ts/"))

        assert exc.value.status == 404

        with pytest.raises(ClientResponseError) as exc:
            http_get(distribution_base_url(f"{distribution.base_url}20259928T092752Z/"))

        assert exc.value.status == 404

    def test_checkpoint_artifact_is_served(self, setup, http_get, checkpoint_url):
        pubs, distribution = setup
        pub_1_url = checkpoint_url(distribution, pubs[1].pulp_created)
        pub_2_url = checkpoint_url(distribution, pubs[2].pulp_created)

        pub_2_response = http_get(f"{pub_2_url}PULP_MANIFEST").decode("utf-8")
        pub_1_response = http_get(f"{pub_1_url}PULP_MANIFEST").decode("utf-8")

        assert pub_2_response == pub_1_response
        lines = pub_1_response.strip().split("\n")
        artifact_names = {line.split(",")[0] for line in lines}
        assert artifact_names == {"0", "1"}

    def test_content_file_is_redirected_to_digest_url(
        self, setup, http_get, checkpoint_url, distribution_base_url
    ):
        pubs, distribution = setup
        pub_1_url = checkpoint_url(distribution, pubs[1].pulp_created)
        arbitrary_url = checkpoint_url(distribution, pubs[1].pulp_created + timedelta(seconds=1))
        pub_3_url = checkpoint_url(distribution, pubs[3].pulp_created)
        distro_base_url = distribution_base_url(distribution.base_url)

        manifest = http_get(f"{pub_1_url}PULP_MANIFEST")
        digest = dict(line.split(",")[:2] for line in manifest.decode("utf-8").split())["0"]
        digest_url = f"{distro_base_url}sha256/{digest}/0"

        # Every checkpoint redirects the same file to the same URL
        for url in (pub_1_url, arbitrary_url, pub_3_url):
            response = requests.get(f"{url}0", allow_redirects=False)
            assert response.status_code == 302
            assert response.headers["Location"] == urlparse(digest_url).path
        assert hashlib.sha256(http_get(digest_url)).hexdigest() == digest

        # Metadata is served from the checkpoint
        response = requests.get(f"{pub_1_url}PULP_MANIFEST", allow_redirects=False)
        assert response.status_code == 200

        # Only files that a checkpoint serves under the requested name are found
        manifest_digest = hashlib.sha256(manifest).hexdigest()
        for path in (f"sha256/{digest}/1", f"sha256/{manifest_digest}/PULP_MANIFEST"):
            with pytest.raises(ClientResponseError) as exc:
                http_get(f"{distro_base_url}{path}")
            assert exc.value.status == 404

    def test_non_checkpoint_timestamp_is_redirected(self, setup, http_get, checkpoint_url):
        pubs, distribution = setup
        # Using a non-checkpoint publication timestamp
        pub_3_url = checkpoint_url(distribution, pubs[3].pulp_created)
        pub_4_url = checkpoint_url(distribution, pubs[4].pulp_created)

        response = http_get(pub_4_url).decode("utf-8")
        assert f"<h1>Index of {urlparse(pub_3_url).path}</h1>" in response

        # Test without a trailing slash
        response = http_get(pub_4_url[:-1]).decode("utf-8")
        assert f"<h1>Index of {urlparse(pub_3_url).path}</h1>" in response

    def test_arbitrary_timestamp_is_redirected(self, setup, http_get, checkpoint_url):
        pubs, distribution = setup
        pub_1_url = checkpoint_url(distribution, pubs[1].pulp_created)
        arbitrary_url = checkpoint_url(distribution, pubs[1].pulp_created + timedelta(seconds=1))

        response = http_get(arbitrary_url).decode("utf-8")
        assert f"<h1>Index of {urlparse(pub_1_url).path}</h1>" in response

        # Test without a trailing slash
        response = http_get(arbitrary_url[:-1]).decode("utf-8")
        assert f"<h1>Index of {urlparse(pub_1_url).path}</h1>" in response

    def test_current_timestamp_serves_latest_checkpoint(self, setup, http_get, checkpoint_url):
        pubs, distribution = setup
        pub_3_url = checkpoint_url(distribution, pubs[3].pulp_created)
        now_url = checkpoint_url(distribution, datetime.now())

        response = http_get(now_url).decode("utf-8")

        assert f"<h1>Index of {urlparse(pub_3_url).path}</h1>" in response

    def test_before_first_timestamp_returns_404(self, setup, http_get, checkpoint_url):
        pubs, distribution = setup
        pub_0_url = checkpoint_url(distribution, pubs[0].pulp_created)

        with pytest.raises(ClientResponseError) as exc:
            http_get(pub_0_url).decode("utf-8")

        assert exc.value.status == 404

    def test_future_timestamp_returns_404(self, setup, http_get, checkpoint_url):
        _, distribution = setup
        url = checkpoint_url(distribution, datetime.now() + timedelta(days=1))

        with pytest.raises(ClientResponseError) as exc:
            http_get(url).decode("utf-8")

        assert exc.value.status == 404

    @pytest.mark.parallel
    def test_checkpoint_publication_with_repository_version_fails(
        self, file_bindings, gen_object_with_cleanup, file_repository_factory
    ):
        """Test that creating checkpoint publication using a repository version fails."""

        repo = file_repository_factory()

        with pytest.raises(exceptions.ApiException):
            gen_object_with_cleanup(
                file_bindings.PublicationsFileApi,
                {"repository_version": repo.latest_version_href, "checkpoint": True},
            )


@pytest.mark.parallel
def test_checkpoint_retention(
    file_bindings,
    file_repository_factory,
    create_publication,
    monitor_task,
):
    """Test retain_checkpoints for repositories.

    When retain_checkpoints is set, only the N most recent checkpoint publications should
    retain their checkpoint=True flag. Older ones get their checkpoint flag cleared.
    """
    repo = file_repository_factory()

    create_publication.precreate(3)
    checkpoint_pubs = [create_publication(repo, True) for _ in range(2)]

    for pub in checkpoint_pubs:
        assert file_bindings.PublicationsFileApi.read(pub.pulp_href).checkpoint is True

    # Set retain_checkpoints=1 — should clear checkpoint flag on the oldest
    task = file_bindings.RepositoriesFileApi.partial_update(
        repo.pulp_href, {"retain_checkpoints": 1}
    ).task
    monitor_task(task)

    assert file_bindings.PublicationsFileApi.read(checkpoint_pubs[0].pulp_href).checkpoint is False
    assert file_bindings.PublicationsFileApi.read(checkpoint_pubs[1].pulp_href).checkpoint is True

    # Create another checkpoint — should trigger steady-state cleanup
    new_pub = create_publication(repo, True)

    assert file_bindings.PublicationsFileApi.read(checkpoint_pubs[1].pulp_href).checkpoint is False
    assert file_bindings.PublicationsFileApi.read(new_pub.pulp_href).checkpoint is True
