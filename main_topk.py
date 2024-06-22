import nltk
from top_k_rag import TopKRAG
from gptj import GPTJTextGenerator


nltk.download('punkt')


paragraphs = [
    "In the beginning was the Word, and the Word was with God, and the Word was God.",
    "He was with God in the beginning.",
    "Through him all things were made; without him nothing was made that has been made.",
    "In him was life, and that life was the light of all mankind.",
    "The light shines in the darkness, and the darkness has not overcome it.",
    "There was a man sent from God whose name was John.",
    "He came as a witness to testify concerning that light, so that through him all might believe.",
    "He himself was not the light; he came only as a witness to the light.",
    "The true light that gives light to everyone was coming into the world.",
    "He was in the world, and though the world was made through him, the world did not recognize him.",
    "He came to that which was his own, but his own did not receive him.",
    "Yet to all who did receive him, to those who believed in his name, he gave the right to become children of God children born not of natural descent, nor of human decision or a husband's will, but born of God.",
    "The Word became flesh and made his dwelling among us. We have seen his glory, the glory of the one and only Son, who came from the Father, full of grace and truth.",
    "Out of his fullness we have all received grace in place of grace already given.",
    "For the law was given through Moses; grace and truth came through Jesus Christ.",
    "No one has ever seen God, but the one and only Son, who is himself God and is in closest relationship with the Father, has made him known.",
    "In the time of Herod king of Judea there was a priest named Zechariah, who belonged to the priestly division of Abijah; his wife Elizabeth was also a descendant of Aaron.",
    "Both of them were righteous in the sight of God, observing all the Lord's commands and decrees blamelessly.",
    "But they were childless because Elizabeth was not able to conceive, and they were both very old.",
    "At that time Mary got ready and hurried to a town in the hill country of Judea, where she entered Zechariah's home and greeted Elizabeth.",
    "When Elizabeth heard Mary's greeting, the baby leaped in her womb, and Elizabeth was filled with the Holy Spirit.",
    "In a loud voice she exclaimed: Blessed are you among women, and blessed is the child you will bear!",
    "But why am I so favored, that the mother of my Lord should come to me?",
    "As soon as the sound of your greeting reached my ears, the baby in my womb leaped for joy.",
    "Blessed is she who has believed that the Lord would fulfill his promises to her!"
]


rag_system = TopKRAG(paragraphs)
text_generator = GPTJTextGenerator()


def get_detailed_answer(question):
    top_k_paragraphs = rag_system.answer_question(question, k=3)
    combined_paragraphs = " ".join(top_k_paragraphs)
    prompt = f"DOCUMENT: {combined_paragraphs} \nQUESTION:{question} \nINSTRUCTIONS: Answer the QUESTION using the DOCUMENT text above. Keep your answer ground in the facts of the DOCUMENT. If the DOCUMENT does not contain the facts to answer the QUESTION return NONE"
    detailed_answer = text_generator.generate_text(prompt)
    return detailed_answer


questions = [
    "What was the name of the priest? What do you know about this person?"
]


for question in questions:
    detailed_answer = get_detailed_answer(question)
    print(f"\nQuestion: {question}")
    print(f"\nDetailed Answer: {detailed_answer}\n")
