from pathlib import Path
import csv
import sqlite3


ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "cell-count.csv"
DATABASE_PATH = ROOT / "cell_counts.db"
POPULATIONS = ("b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte")
REQUIRED_COLUMNS = {
    "project",
    "subject",
    "condition",
    "age",
    "sex",
    "treatment",
    "response",
    "sample",
    "sample_type",
    "time_from_treatment_start",
    *POPULATIONS,
}


def load_data() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {CSV_PATH}")

    with CSV_PATH.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        columns = set(reader.fieldnames or [])
        missing_columns = REQUIRED_COLUMNS - columns
        if missing_columns:
            raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
        rows = list(reader)

    sample_ids = [row["sample"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("The sample column must contain unique sample IDs")
    if not rows:
        raise ValueError("The input CSV contains no data rows")

    for row_number, row in enumerate(rows, start=2):
        try:
            counts = [int(row[population]) for population in POPULATIONS]
            age = int(row["age"])
            timepoint = int(row["time_from_treatment_start"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid numeric value on CSV row {row_number}") from error
        if age < 0 or timepoint < 0 or any(count < 0 for count in counts):
            raise ValueError(f"Negative numeric value on CSV row {row_number}")
        if sum(counts) == 0:
            raise ValueError(f"All population counts are zero on CSV row {row_number}")

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            DROP VIEW IF EXISTS sample_population_summary;
            DROP TABLE IF EXISTS cell_counts;
            DROP TABLE IF EXISTS samples;

            CREATE TABLE samples (
                sample_id TEXT PRIMARY KEY,
                project TEXT NOT NULL,
                subject_id TEXT NOT NULL,
                condition TEXT NOT NULL,
                age INTEGER,
                sex TEXT,
                treatment TEXT NOT NULL,
                response TEXT,
                sample_type TEXT NOT NULL,
                time_from_treatment_start INTEGER NOT NULL
            );

            CREATE TABLE cell_counts (
                sample_id TEXT NOT NULL REFERENCES samples(sample_id),
                population TEXT NOT NULL,
                count INTEGER NOT NULL CHECK (count >= 0),
                PRIMARY KEY (sample_id, population)
            );

            CREATE VIEW sample_population_summary AS
            WITH totals AS (
                SELECT sample_id, SUM(count) AS total_count
                FROM cell_counts
                GROUP BY sample_id
            )
            SELECT
                s.sample_id AS sample,
                t.total_count,
                c.population,
                c.count,
                ROUND(100.0 * c.count / t.total_count, 6) AS percentage,
                s.project,
                s.subject_id,
                s.condition,
                s.age,
                s.sex,
                s.treatment,
                s.response,
                s.sample_type,
                s.time_from_treatment_start
            FROM samples AS s
            JOIN totals AS t ON t.sample_id = s.sample_id
            JOIN cell_counts AS c ON c.sample_id = s.sample_id;
            """
        )

        sample_records = []
        count_records = []
        for row in rows:
            sample_id = row["sample"]
            sample_records.append(
                (
                    sample_id,
                    row["project"],
                    row["subject"],
                    row["condition"],
                    int(row["age"]),
                    row["sex"],
                    row["treatment"],
                    row["response"] or None,
                    row["sample_type"],
                    int(row["time_from_treatment_start"]),
                )
            )
            count_records.extend(
                (sample_id, population, int(row[population]))
                for population in POPULATIONS
            )

        connection.executemany(
            """
            INSERT INTO samples (
                sample_id, project, subject_id, condition, age, sex, treatment,
                response, sample_type, time_from_treatment_start
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            sample_records,
        )
        connection.executemany(
            "INSERT INTO cell_counts (sample_id, population, count) VALUES (?, ?, ?)",
            count_records,
        )

    print(f"Loaded {len(sample_records):,} samples into {DATABASE_PATH.name}")


if __name__ == "__main__":
    load_data()