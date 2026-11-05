from normalize_coaches import normalize_coaches
from normalize_teams import normalize_teams


def main():
    print("Normalizing teams...")
    normalize_teams()

    print("Normalizing coaches...")
    normalize_coaches()

    print("Normalization completed.")


if __name__ == "__main__":
    main()