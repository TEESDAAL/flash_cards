import sys
from random import shuffle

type Catagory = str


def select_mode(flashcards: dict[Catagory, list[tuple[str, str]]]):
    print("What catagory do you want to study? (Default is to practice all of them)")
    print('"": All questions.')
    print(*[f"{i}: {catagory}" for i, catagory in enumerate(flashcards.keys(), start=1)], sep="\n")
    print("Q: quit")
    options = {str(i+1) for i in range(len(flashcards.keys()))} | {"", "Q"}
    while (catagory := input("")) not in options:
        print(f"Please input one of {list(sorted(options))}")

    if catagory == "Q":
        exit(0)

    if catagory == "":
        return ask_questions([question for questions in flashcards.values() for question in questions])

    return ask_questions({str(i+1): questions for i, questions in enumerate(flashcards.values())}[catagory])


def ask_questions(questions: list[tuple[str, str]]):
    shuffle(questions)
    for i, (question, answer) in enumerate(questions, start=1):
        input(f"{i}/{len(questions)}: {question}\n")
        print(answer)
        input("(y/n): ")


if __name__ == "__main__":
    flashcards = __import__(sys.argv[1]).flashcards
    while True:
        select_mode(flashcards)

