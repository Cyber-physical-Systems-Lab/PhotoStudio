"""Spoken prompts. Keys are used by the session; add languages by adding a column (en, sv, zh)."""

PHRASES = {
    'greeting': {
        'en': 'Hello {name}! Welcome to the lab. I am the lab photographer. Please sit down comfortably and look towards the camera.',
        'sv': 'Hej {name}! Välkommen till labbet. Jag är labbets fotograf. Sätt dig bekvämt och titta mot kameran.',
        'zh': '你好，{name}！欢迎来到实验室。我是实验室的摄影师。请舒服地坐下，看向镜头。',
    },
    'searching': {
        'en': 'I am looking for you. Please sit on the chair, facing me.',
        'sv': 'Jag letar efter dig. Sätt dig på stolen, vänd mot mig.',
        'zh': '我正在找你。请坐在椅子上，面向我。',
    },
    'found': {
        'en': 'I can see you. I will move the camera into position. Please stay where you are.',
        'sv': 'Nu ser jag dig. Jag flyttar kameran. Stå kvar där du är.',
        'zh': '我看到你了。我会把相机移到合适的位置，请保持不动。',
    },
    'expr_big_smile': {
        'en': 'Now a big, happy smile!',
        'sv': 'Nu ett stort, glatt leende!',
        'zh': '现在，来一个开心的大笑脸！',
    },
    'expr_thumbs_up': {
        'en': 'Give me a thumbs up, next to your face!',
        'sv': 'Visa tummen upp, bredvid ansiktet!',
        'zh': '在脸旁边竖起大拇指吧！',
    },
    'expr_serious': {
        'en': 'Now a calm, confident look. No need to smile.',
        'sv': 'Nu en lugn, självsäker blick. Du behöver inte le.',
        'zh': '现在来一个平静、自信的表情，不用笑。',
    },
    'expr_laugh': {
        'en': 'Think of something funny, and laugh a little.',
        'sv': 'Tänk på något roligt, och skratta lite.',
        'zh': '想一件好笑的事，轻轻笑一下。',
    },
    'expr_arms_crossed': {
        'en': 'Cross your arms and look proud.',
        'sv': 'Korsa armarna och se stolt ut.',
        'zh': '双臂交叉，显得自豪一点。',
    },
    'expr_look_away': {
        'en': 'Look a little past the camera, as if you are thinking.',
        'sv': 'Titta lite förbi kameran, som om du tänker.',
        'zh': '视线稍微越过镜头，像在思考一样。',
    },
    'style_natural': {
        'en': 'We will take some relaxed, natural photos. Sit comfortably, be yourself, and have fun.',
        'sv': 'Vi tar några avslappnade, naturliga bilder. Sitt bekvämt, var dig själv och ha kul.',
        'zh': '我们来拍几张轻松自然的照片。舒服地坐着，做你自己，开心一点。',
    },
    'style_professional': {
        'en': 'We will take a professional portrait. Sit up straight, relax your shoulders, and bring your chin slightly forward and down.',
        'sv': 'Vi tar ett professionellt porträtt. Sitt rakt, slappna av i axlarna och för hakan lite framåt och nedåt.',
        'zh': '我们来拍一张专业肖像。请坐直，放松肩膀，下巴稍微向前、向下一点。',
    },
    'expr_confident_smile': {
        'en': 'A confident, gentle smile, with your lips closed.',
        'sv': 'Ett självsäkert, lätt leende, med stängd mun.',
        'zh': '一个自信、温和的微笑，嘴唇闭合。',
    },
    'sit_down': {
        'en': 'Please sit down on the chair. The photos are taken seated.',
        'sv': 'Sätt dig på stolen, är du snäll. Bilderna tas sittande.',
        'zh': '请坐到椅子上。我们坐着拍照。',
    },
    'sit_up': {
        'en': 'Please sit up a little straighter.',
        'sv': 'Sitt lite rakare, är du snäll.',
        'zh': '请坐得再直一点。',
    },
    'hold_marker': {
        'en': 'Please hold the marker cube in front of your eyes, facing the camera.',
        'sv': 'Håll markörkuben framför ögonen, vänd mot kameran.',
        'zh': '请把标记方块举在眼睛前面，正对镜头。',
    },
    'marker_done': {
        'en': 'Thank you. You can put the marker down.',
        'sv': 'Tack. Du kan lägga ner markören.',
        'zh': '谢谢，可以把标记放下了。',
    },
    'marker_missed': {
        'en': 'I could not see the marker. That is fine, please put it down.',
        'sv': 'Jag såg inte markören. Det gör inget, lägg ner den.',
        'zh': '我没有看到标记。没关系，请把它放下。',
    },
    'turn_left': {
        'en': 'Please turn your head a little to your left.',
        'sv': 'Vrid huvudet lite åt vänster.',
        'zh': '请把头稍微向左转一点。',
    },
    'turn_right': {
        'en': 'Please turn your head a little to your right.',
        'sv': 'Vrid huvudet lite åt höger.',
        'zh': '请把头稍微向右转一点。',
    },
    'step_back': {
        'en': 'Please move back a little.',
        'sv': 'Flytta dig lite bakåt.',
        'zh': '请稍微往后一点。',
    },
    'step_forward': {
        'en': 'Please come a little closer.',
        'sv': 'Kom lite närmare.',
        'zh': '请稍微靠近一点。',
    },
    'step_left': {
        'en': 'Please move a little to your left.',
        'sv': 'Flytta dig lite åt vänster.',
        'zh': '请稍微往左边移一点。',
    },
    'step_right': {
        'en': 'Please move a little to your right.',
        'sv': 'Flytta dig lite åt höger.',
        'zh': '请稍微往右边移一点。',
    },
    'chin_down': {
        'en': 'Chin down slightly, please.',
        'sv': 'Sänk hakan lite.',
        'zh': '请把下巴稍微收一点。',
    },
    'chin_up': {
        'en': 'Chin up slightly, please.',
        'sv': 'Höj hakan lite.',
        'zh': '请把下巴稍微抬一点。',
    },
    'level_head': {
        'en': 'Please keep your head level.',
        'sv': 'Håll huvudet rakt.',
        'zh': '请保持头部端正。',
    },
    'perfect': {
        'en': 'Perfect. Hold that.',
        'sv': 'Perfekt. Stanna så.',
        'zh': '很好，保持住。',
    },
    'look_at_camera': {
        'en': 'Look into the camera lens and relax your shoulders.',
        'sv': 'Titta in i kameran och slappna av i axlarna.',
        'zh': '请看着镜头，放松肩膀。',
    },
    'smile': {
        'en': 'Give me a natural smile.',
        'sv': 'Ge mig ett naturligt leende.',
        'zh': '自然地微笑一下。',
    },
    'countdown': {
        'en': 'Three. Two. One.',
        'sv': 'Tre. Två. Ett.',
        'zh': '三，二，一。',
    },
    'next_angle': {
        'en': 'Nice. One more angle. Keep your head where it is.',
        'sv': 'Fint. En vinkel till. Håll huvudet där det är.',
        'zh': '很好。再换一个角度，头保持不动。',
    },
    'lost': {
        'en': 'I lost sight of you. Please sit back on the chair.',
        'sv': 'Jag ser dig inte längre. Sätt dig på stolen igen.',
        'zh': '我看不到你了。请坐回椅子上。',
    },
    'done': {
        'en': 'All done, thank you {name}! Your portrait is being prepared.',
        'sv': 'Klart, tack {name}! Ditt porträtt förbereds nu.',
        'zh': '拍好了，谢谢你，{name}！你的照片正在处理中。',
    },
    'aborted': {
        'en': 'Sorry, I have to stop the session. Please ask a lab member for help.',
        'sv': 'Förlåt, jag måste avbryta. Be någon i labbet om hjälp.',
        'zh': '抱歉，我需要停止拍摄。请找实验室的工作人员帮忙。',
    },
}

def phrase(key: str, lang: str = 'en', **kw) -> str:
    table = PHRASES.get(key)
    if table is None:
        return key
    text = table.get(lang) or table['en']
    try:
        return text.format(**kw)
    except KeyError:
        return text.replace('{name}', '')
