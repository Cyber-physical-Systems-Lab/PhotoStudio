import os
import sys

from portrait_session.styles import SESSION_STYLES

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'portrait_voice'))
from portrait_voice.phrases import PHRASES  # noqa: E402


def test_presets_are_consistent_and_spoken_in_every_language():
    for name, p in SESSION_STYLES.items():
        assert p['style'] == name
        assert len(p['shot_azimuths_deg']) == len(p['shot_counts'])
        assert all(a < 0 for a in p['shot_azimuths_deg'])      # subject's left keeps the wrist clear
        for key in p['expressions'] + [f'style_{name}']:
            assert key in PHRASES, key
            assert all(PHRASES[key].get(lang) for lang in ('en', 'sv', 'zh')), key
