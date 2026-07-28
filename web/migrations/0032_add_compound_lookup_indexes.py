from django.db import migrations


def add_lookup_indexes(apps, schema_editor):
    table_name = "web_compoundlibrary"
    indexes = {
        "web_compound_sample_match_idx": (
            "spectrum_type",
            "matched_spectrum_id",
            "ionmode",
        ),
        "web_compound_standard_match_idx": (
            "spectrum_type",
            "standard_id",
            "ionmode",
        ),
    }

    with schema_editor.connection.cursor() as cursor:
        constraints = schema_editor.connection.introspection.get_constraints(
            cursor, table_name
        )
        for index_name, columns in indexes.items():
            if index_name in constraints:
                continue
            quoted_columns = ", ".join(
                schema_editor.quote_name(column) for column in columns
            )
            cursor.execute(
                f"CREATE INDEX {schema_editor.quote_name(index_name)} "
                f"ON {schema_editor.quote_name(table_name)} ({quoted_columns})"
            )


class Migration(migrations.Migration):
    dependencies = [
        ("web", "0031_add_missing_mw_antitumor_columns"),
    ]

    operations = [
        migrations.RunPython(
            add_lookup_indexes,
            reverse_code=migrations.RunPython.noop,
        ),
    ]
