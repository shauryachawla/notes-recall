from data_cleaning.ingestion import fetch_notes

def main():
    """Fetch notes from the Keep resource and print the count."""
    notes = fetch_notes('resources/Keep')
    print(f"Fetched {len(notes)} notes.")

if __name__ == "__main__":
    main()
