# Generated by Django 4.2.13 on 2024-07-23 13:03

from django.db import migrations, models
import django.db.models.deletion
import django_lifecycle.mixins
import pulpcore.app.models.base


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0120_get_url_removal'),
    ]

    operations = [
        migrations.AlterModelOptions(
            name='task',
            options={'permissions': [('manage_roles_task', 'Can manage role assignments on task'), ('view_task_profile_artifacts', 'Can view profile data for task')]},
        ),
        migrations.CreateModel(
            name='ProfileArtifact',
            fields=[
                ('pulp_id', models.UUIDField(default=pulpcore.app.models.base.pulp_uuid, editable=False, primary_key=True, serialize=False)),
                ('pulp_created', models.DateTimeField(auto_now_add=True)),
                ('pulp_last_updated', models.DateTimeField(auto_now=True, null=True)),
                ('name', models.TextField()),
                ('artifact', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='core.artifact')),
                ('task', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='core.task')),
            ],
            options={
                'unique_together': {('task', 'name')},
            },
            bases=(django_lifecycle.mixins.LifecycleModelMixin, models.Model),
        ),
        migrations.AddField(
            model_name='task',
            name='profile_artifacts',
            field=models.ManyToManyField(through='core.ProfileArtifact', to='core.artifact'),
        ),
    ]