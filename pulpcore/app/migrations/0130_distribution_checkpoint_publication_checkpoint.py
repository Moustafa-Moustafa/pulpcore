# Generated by Django 4.2.19 on 2025-02-26 11:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("core", "0129_content_pulp_labels"),
    ]

    operations = [
        migrations.AddField(
            model_name="distribution",
            name="checkpoint",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="publication",
            name="checkpoint",
            field=models.BooleanField(default=False),
        ),
    ]
