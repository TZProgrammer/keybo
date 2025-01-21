from optimizer import Optimizer
from classifier import Keyboard
from scorer import FreyaScorer

keyboard_chars = "qwertyuiopasdfghjkl'zxcvbnm,.-"

def generate(num_attempts):
    # Optimize
    best_kb = None
    best_score = float("inf")

    for _ in range(num_attempts):
        keyboard = Keyboard(keyboard_chars)
        scorer = FreyaScorer()
        optimizer = Optimizer(keyboard, scorer)
        optimizer.optimize(keyboard, scorer)

        print("Fitness", int(optimizer.fitness))

        if optimizer.fitness < best_score:
            best_score = optimizer.fitness
            print("new best")
            best_kb = optimizer.keyboard

            with open("logfile.txt", "a") as f:
                # Write to the file
                f.write(f"{best_score} @{optimizer.tg_coverage}\n")
                f.write(repr(best_kb) + "\n")

    print("best score")
    print(best_kb)

def main():
    num_attempts = 10
    generate(num_attempts=num_attempts)

if __name__ == "__main__":
    main()
