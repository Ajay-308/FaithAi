"""
Scripture utilities — Bible book validation, denomination helpers,
verse reference extraction.
No AI calls — pure data / regex.
"""

import re


#  Bible books ─

BIBLE_BOOKS = {
    "genesis", "exodus", "leviticus", "numbers", "deuteronomy",
    "joshua", "judges", "ruth", "1 samuel", "2 samuel",
    "1 kings", "2 kings", "1 chronicles", "2 chronicles",
    "ezra", "nehemiah", "esther", "job", "psalm", "psalms",
    "proverbs", "ecclesiastes", "song of solomon", "isaiah",
    "jeremiah", "lamentations", "ezekiel", "daniel",
    "hosea", "joel", "amos", "obadiah", "jonah", "micah",
    "nahum", "habakkuk", "zephaniah", "haggai", "zechariah",
    "malachi",
    "matthew", "mark", "luke", "john", "acts", "romans",
    "1 corinthians", "2 corinthians", "galatians", "ephesians",
    "philippians", "colossians", "1 thessalonians",
    "2 thessalonians", "1 timothy", "2 timothy", "titus",
    "philemon", "hebrews", "james", "1 peter", "2 peter",
    "1 john", "2 john", "3 john", "jude", "revelation",
}


#  Denomination helpers 

def get_translation_for_denomination(denomination: str = "general") -> str:
    mapping = {
        "general":   "NIV",
        "catholic":  "NABRE",
        "orthodox":  "OSB",
        "baptist":   "ESV",
        "reformed":  "ESV",
        "protestant":"NIV",
        "lutheran":  "NRSV",
    }
    return mapping.get(denomination.lower(), "NIV")


def get_denomination_context(denomination: str = "general") -> str:
    contexts = {
        "general": (
            "Non-denominational Christian perspective. Draw on the broad evangelical tradition, "
            "acknowledging diversity within the body of Christ."
        ),
        "protestant": (
            "Protestant theology emphasising Sola Scriptura, Sola Fide, and the priesthood "
            "of all believers. Reformation heritage (Luther, Calvin, Zwingli)."
        ),
        "catholic": (
            "Catholic theology emphasising Scripture AND Tradition, the Magisterium, "
            "sacramental life, the Communion of Saints, and papal authority. "
            "Deuterocanonical books included in canon."
        ),
        "orthodox": (
            "Eastern Orthodox theology emphasising theosis, liturgy, the Church Fathers, "
            "the seven Ecumenical Councils, and the icon tradition. "
            "Septuagint-based canon. Trinity understood via Cappadocian theology."
        ),
        "baptist": (
            "Evangelical Baptist perspective: Sola Scriptura, believer's baptism by immersion, "
            "congregational church governance, symbolic view of communion."
        ),
        "reformed": (
            "Reformed/Calvinist theology: TULIP (Total depravity, Unconditional election, "
            "Limited atonement, Irresistible grace, Perseverance of saints), "
            "covenant theology, Westminster Confession."
        ),
        "lutheran": (
            "Lutheran theology: Law and Gospel distinction, sacramental real presence, "
            "two-kingdoms doctrine, Augsburg Confession heritage."
        ),
    }
    return contexts.get(denomination.lower(), contexts["general"])


#  Verse reference extraction 

def extract_verse_refs(text: str) -> list[str]:
    """
    Extract Bible verse references from free text.
    Matches patterns like: John 3:16, 1 Corinthians 13:4, Rev. 22:21
    """
    pattern = r'([1-3]?\s?[A-Za-z]+(?:\s[A-Za-z]+)?)\s+(\d+):(\d+)'
    matches = re.findall(pattern, text)

    refs = []
    for book, chapter, verse in matches:
        book_clean = book.strip()
        if book_clean.lower() in BIBLE_BOOKS:
            refs.append(f"{book_clean} {chapter}:{verse}")

    return refs


#  Verse structure validation 

def validate_verse_ref(book: str, chapter: int, verse: int) -> tuple[bool, str]:
    """
    Structural validation: is this book name real, and are chapter/verse positive?
    Does NOT check max-chapter or max-verse limits (that requires a full Bible data set).

    Returns:
        (True, book_name)  if valid
        (False, error_msg) if invalid
    """
    if book.lower().strip() not in BIBLE_BOOKS:
        return False, f"'{book}' is not a recognised Bible book."

    if chapter <= 0 or verse <= 0:
        return False, "Chapter and verse must be positive numbers."

    return True, book