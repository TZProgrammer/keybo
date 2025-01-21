from optimizer import Optimizer

def generate(num_attempts):
    # Optimize
    best_kb = None
    best_score = float("inf")

    for _ in range(num_attempts):
        optimizer = Optimizer()
        optimizer.optimize()

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
