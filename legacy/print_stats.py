from classifier import Keyboard
from scorer import BigramXGBoostScorer
from wpm_conditioned_model import TypingModel

def get_qwerty_score(wpm_base):
    qwerty_chars = "qwertyuiopasdfghjkl;zxcvbnm,./"
    keyboard = Keyboard(qwerty_chars)
    scorer = BigramXGBoostScorer(target_wpm=wpm_base)
    return scorer.get_fitness(keyboard)

def get_dvorak_score(wpm_base):
    dvorak_chars = "',.pyfgcrlaoeuidhtns;qjkxbmwvz"
    keyboard = Keyboard(dvorak_chars)
    scorer = BigramXGBoostScorer(target_wpm=wpm_base)
    return scorer.get_fitness(keyboard)

def get_graphite_score(wpm_base):
    graphite_chars = "bldwz-foujnrtsgyhaei,qxmcvkp.'"
    keyboard = Keyboard(graphite_chars)
    scorer = BigramXGBoostScorer(target_wpm=wpm_base)
    return scorer.get_fitness(keyboard)

def get_pinev4_score(wpm_base):
    pinev4_chars = "qlcmk'fuoynrstwpheaijxzgvbd,.-"
    keyboard = Keyboard(pinev4_chars)
    scorer = BigramXGBoostScorer(target_wpm=wpm_base)
    return scorer.get_fitness(keyboard)

def get_semimak_score(wpm_base):
    semimak_chars = "flhvzqwuoysrntkcdeaix'bmjpg,.-"
    keyboard = Keyboard(semimak_chars)
    scorer = BigramXGBoostScorer(target_wpm=wpm_base)
    return scorer.get_fitness(keyboard)

def get_semimakJQ_score(wpm_base):
    semimak_chars = "flhvz'wuoysrntkcdeaixjbmqpg,.-"
    keyboard = Keyboard(semimak_chars)
    scorer = BigramXGBoostScorer(target_wpm=wpm_base)
    return scorer.get_fitness(keyboard)

def get_tusk_score(wpm_base):
    tusk_chars = "tskcwzqioumvypjnreaxlfgdbh-'.,"
    keyboard = Keyboard(tusk_chars)
    scorer = BigramXGBoostScorer(target_wpm=wpm_base)
    return scorer.get_fitness(keyboard)

def get_threa_score(wpm_base):
    tusk_chars = "cywbjlquo.fspkthrea,vgdmxnzi'-"
    keyboard = Keyboard(tusk_chars)
    scorer = BigramXGBoostScorer(target_wpm=wpm_base)
    return scorer.get_fitness(keyboard)

def print_layout_scores(wpm_base):
    # Compute qwerty score as baseline.
    qwerty_score = get_qwerty_score(wpm_base)
    print(f"qwerty score: {qwerty_score} (baseline)")

    # Define the other layouts to compare.
    layouts = [
        ("dvorak", get_dvorak_score),
        ("graphite", get_graphite_score),
        ("pinev4", get_pinev4_score),
        ("semimak", get_semimak_score),
        ("semimak-JQ", get_semimakJQ_score),
        ("tusk", get_tusk_score),
        ("threa", get_threa_score)
    ]

    # Print each layout's score and the improvement over qwerty.
    for layout_name, score_func in layouts:
        score = score_func(wpm_base)
        # Calculate percentage improvement over qwerty.
        improvement_pct = ((qwerty_score - score) / qwerty_score) * 100
        print(f"{layout_name} score: {score} (Improvement over qwerty: {improvement_pct:+.2f}%)")

def main():
    WPM_BASE = 100
    print_layout_scores(WPM_BASE)

if __name__ == "__main__":
    main()
