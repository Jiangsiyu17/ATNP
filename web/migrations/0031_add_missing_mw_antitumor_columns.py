from django.db import migrations


def add_missing_columns(apps, schema_editor):
    """Repair columns declared in the replaced initial migration but absent in the old DB."""
    table_name = "web_compoundlibrary"

    with schema_editor.connection.cursor() as cursor:
        existing_columns = {
            column.name
            for column in schema_editor.connection.introspection.get_table_description(
                cursor, table_name
            )
        }

        if "mw" not in existing_columns:
            cursor.execute(
                f"ALTER TABLE {table_name} ADD COLUMN mw DOUBLE NULL"
            )

        if "antitumor" not in existing_columns:
            cursor.execute(
                f"ALTER TABLE {table_name} "
                "ADD COLUMN antitumor TINYINT(1) NOT NULL DEFAULT 0"
            )

        constraints = schema_editor.connection.introspection.get_constraints(
            cursor, table_name
        )

        if "web_compoundlibrary_mw_9c334908" not in constraints:
            cursor.execute(
                f"CREATE INDEX web_compoundlibrary_mw_9c334908 "
                f"ON {table_name} (mw)"
            )

        if "web_compoundlibrary_antitumor_ba89097d" not in constraints:
            cursor.execute(
                f"CREATE INDEX web_compoundlibrary_antitumor_ba89097d "
                f"ON {table_name} (antitumor)"
            )


class Migration(migrations.Migration):
    dependencies = [
        ("web", "0030_compoundlibrary_inchikey"),
    ]

    operations = [
        migrations.RunPython(
            add_missing_columns,
            reverse_code=migrations.RunPython.noop,
        ),
    ]
