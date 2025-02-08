from classifier import Keyboard
from scorer import BigramXGBoostScorer, data_size
from wpm_conditioned_model import TypingModel


def get_qwerty_score(wpm_base):
    qwerty_chars = "qwertyuiopasdfghjkl'zxcvbnm,.-"

    keyboard = Keyboard(data_size, qwerty_chars)
    scorer = BigramXGBoostScorer(target_wpm=wpm_base)

    return scorer.get_fitness(keyboard)

def get_dvorak_score(wpm_base):
    dvorak_chars = "',.pyfgcrlaoeuidhtns;qjkxbmwvz"

    keyboard = Keyboard(data_size, dvorak_chars)
    scorer = BigramXGBoostScorer(target_wpm=wpm_base)

    return scorer.get_fitness(keyboard)

def get_graphite_score(wpm_base):
    graphite_chars = "bldwz-foujnrtsgyhaei,qxmcvkp.'"

    keyboard = Keyboard(data_size, graphite_chars)
    scorer = BigramXGBoostScorer(target_wpm=wpm_base)

    return scorer.get_fitness(keyboard)

def get_pinev4_score(wpm_base):
    pinev4_chars = "qlcmk'fuoynrstwpheaijxzgvbd,.-"

    keyboard = Keyboard(data_size, pinev4_chars)
    scorer = BigramXGBoostScorer(target_wpm=wpm_base)

    return scorer.get_fitness(keyboard)

def get_semimak_score(wpm_base):
    semimak_chars = "flhvzqwuoysrntkcdeaix'bmjpg,.-"

    keyboard = Keyboard(data_size, semimak_chars)
    scorer = BigramXGBoostScorer(target_wpm=wpm_base)

    return scorer.get_fitness(keyboard)

def get_semimakJQ_score(wpm_base):
    semimak_chars = "flhvz'wuoysrntkcdeaixjbmqpg,.-"

    keyboard = Keyboard(data_size, semimak_chars)
    scorer = BigramXGBoostScorer(target_wpm=wpm_base)

    return scorer.get_fitness(keyboard)

def print_qwerty_score(wpm_base):
    score = get_qwerty_score(wpm_base)
    print("qwerty score:", score)

def print_dvorak_score(wpm_base):
    score = get_dvorak_score(wpm_base)
    print("dvorak score:", score)

def print_graphite_score(wpm_base):
    score = get_graphite_score(wpm_base)
    print("graphite score:", score)

def print_pinev4_score(wpm_base):
    score = get_pinev4_score(wpm_base)
    print("pinev4 score:", score)

def print_semimak_score(wpm_base):
    score = get_semimak_score(wpm_base)
    print("semimak score:", score)

def print_semimakJQ_score(wpm_base):
    score = get_semimakJQ_score(wpm_base)
    print("semimak-JQ score:", score)

def print_layout_scores(wpm_base):
    print_qwerty_score(wpm_base)
    print_dvorak_score(wpm_base)
    print_graphite_score(wpm_base)
    print_pinev4_score(wpm_base)
    print_semimak_score(wpm_base)
    print_semimakJQ_score(wpm_base)

def main():
    WPM_BASE = 100
    print_layout_scores(WPM_BASE)

if __name__ == "__main__":
    main()
