from portrait_voice.phrases import PHRASES, phrase


def test_every_phrase_has_english_swedish_and_chinese():
    for key, t in PHRASES.items():
        assert t.get('en') and t.get('sv') and t.get('zh'), key


def test_name_substitution():
    assert 'Ada' in phrase('greeting', 'en', name='Ada')
    assert phrase('greeting', 'sv', name='Ada').startswith('Hej Ada')
    assert phrase('unknown_key') == 'unknown_key'


def test_chinese_name_substitution():
    assert 'Ada' in phrase('greeting', 'zh', name='Ada')
