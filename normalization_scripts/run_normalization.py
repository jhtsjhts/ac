from normalize_coaches import normalize_coaches
from normalize_goalies import normalize_goalies
from normalize_players import normalize_players
from normalize_teams import normalize_teams


def main():
    print("Normalizing teams...")
    normalize_teams()

    print("Normalizing players...")
    normalize_players()

    print("Normalizing goalies...")
    normalize_goalies()

    print("Normalizing coaches...")
    normalize_coaches()

    print("Normalization completed.")


if __name__ == "__main__":
    main()