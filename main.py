from data_cleaning.ingestion import fetch_notes

def main():
    print("Hello from notes-recall!")
    notes = fetch_notes('resources/Keep')
    print(f"Fetched {len(notes)} notes.")

if __name__ == "__main__":
    main()
