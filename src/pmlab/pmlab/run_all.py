from . import config
from . import db
from .train import train_models_from_db
from .generate_synthetic import generate_synthetic_stream
from .detect import detect_events_on_stream, save_events_to_db
from .visualize import plot_events

def main():
    config.ensure_dirs()
    eng = db.engine()

    db.create_tables(eng)

    # Clean tables for repeatable runs (optional but helps)
    db.clear_table(eng, config.TRAINING_TABLE)
    db.clear_table(eng, config.STREAMING_TABLE)

    # 1) Upload training CSV -> Neon
    train_df = config.load_training_csv(config.DATA_RAW)
    db.upload_df(train_df, eng, config.TRAINING_TABLE)

    # 2) Train models by pulling from Neon
    train_models_from_db(eng)

    # 3) Generate synthetic stream
    stream_df = generate_synthetic_stream(eng, n_points=2000)

    # 4) Upload stream to Neon (proves streaming ingestion)
    db.upload_df(stream_df[[config.DB_TIME] + config.DB_AXIS_COLS], eng, config.STREAMING_TABLE)

    # 5) Detect events
    events_df = detect_events_on_stream(stream_df)

    # 6) Store events in Neon
    save_events_to_db(eng, events_df)

    # 7) Plot
    plot_events(stream_df, events_df)

    print("DONE.")
    print("Check outputs/: thresholds.json, events.csv, synthetic_stream.csv")
    print("Check outputs/plots for axis*_events.png")

if __name__ == "__main__":
    main()
