# Generated by Django 4.2.17 on 2024-12-13 15:11

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0126_remoteartifact_failed_at"),
    ]

    operations = [
        migrations.AddField(
            model_name="distribution",
            name="snapshot",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="publication",
            name="snapshot",
            field=models.BooleanField(default=False),
        ),
    ]