from optimizer import Optimizer
from classifier import Keyboard
from scorer import BigramXGBoostScorer
from wpm_conditioned_model import TypingModel

qwerty = "qwertyuiopasdfghjkl'zxcvbnm,.-"
tusk_1 = "tskcwjqioumvypznreaxlfgdbh-'.,"
tusk_2 = "tskcwzqioumvypjnreaxlfgdbh-'.,"
threa  = "cywbjlquo.fspkthrea,vgdmxnzi'-"
test   = "dvgcwhreoukmybjnzia'tfpsxlq-,."

def generate():
    layout = test
    keyboard = Keyboard(layout)
    scorer = BigramXGBoostScorer()
    optimizer = Optimizer(keyboard, scorer)
    optimizer.local_improvement_2opt(keyboard, scorer)
    print("Fitness", optimizer.best_fitness)
    print(optimizer.best_keyboard)


def main():
    generate()

if __name__ == "__main__":
    main()
