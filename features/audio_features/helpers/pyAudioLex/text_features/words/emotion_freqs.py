'''
@package: pyAudioLex
@author: Jim Schwoebel
@module: emotion_freqs

Take in a text sample and output a range of emotional frequencies.

The script does this for looking for specific hotwords related to the 7 main emotions:
fear, anger, sadness, joy, disgust, surprise, trust, and anticipation.

Note we will train these hotwords on actual emotions into the future and make this
hotword detection more accurate.
'''

from nltk import word_tokenize

def emotion_freqs(importtext):
    
    tokens=word_tokenize(importtext)

    #emotions - fear, anger, sadness, joy, disgust, suprise, trust, anticipation
    fearwords=['scared','afraid','avoid','not','no','anxiety','road','spider','snake','heights','die','falling','death','fast','despair','agonize','bother','worry','endure','sustain','tolerate','creeps','jitters','nervous','nervousness','concerned','worry']
    angerwords=['angry','mad','injustice','annoyed','school','work','predictable','upset','frustrated','sick','tired','fuck','shoot','shit','darn','sucks','bad','ugly']
    sadwords=['sad','depressed','cry','bad','disappointed','distress','uneasy','upset','regret','dismal','black','hopeless']
    joywords=['happy','glad','swell','pleasant','well','good','joy','sweet','grateful','ecstatic','euphoric','encouraged','smile','laugh','content','satisfied','delighted']
    disgustwords=['wrong','disgusting','bad','taste', 'aversion','horror','repulsed','hate','allergy','dislike','displeasure']
    surprisewords=['surprised','appetite','fondness','like','relish','shine','surprise','unexpected','random','new','plastic','cool']
    trustwords=['useful','trust','listen','insight','believe','seek','see','feel','touch','mom','brother','friend','girlfriend','father','dad','uncle','family']
    anticipationwords=['excited','looking','forward','to','birthday','anniversary','christmas','new years','halloween','party','expectation']
    fear=0
    anger=0
    sad=0
    joy=0
    disgust=0
    surprise=0
    trust=0
    anticipation=0
    for i in range(len(tokens)):
        if tokens[i].lower() in fearwords:
            fear=fear+1
        if tokens[i].lower() in angerwords:
            anger=anger+1
        if tokens[i].lower() in sadwords:
            sad=sad+1
        if tokens[i].lower() in joywords:
            joy=joy+1
        if tokens[i].lower() in disgustwords:
            disgust=digust+1
        if tokens[i].lower() in surprisewords:
            surprise=surprise+1
        if tokens[i].lower() in trustwords:
            trust=trust+1
        if tokens[i].lower() in anticipationwords:
            anticipation=anticipation+1

    fearfreq=float(fear/len(tokens))
    angerfreq=float(anger/len(tokens))
    sadfreq=float(sad/len(tokens))
    joyfreq=float(joy/len(tokens))
    disgustfreq=float(disgust/len(tokens))
    surprisefreq=float(surprise/len(tokens))
    trustfreq=float(trust/len(tokens))
    anticipationfreq=float(anticipation/len(tokens))
    
    return [fearfreq,angerfreq,sadfreq,joyfreq,disgustfreq,surprisefreq,trustfreq,anticipationfreq]

#test below 
#print(emotion_freqs('this is a test of sad emotional detection freqs'))
        #[0.0, 0.0, 0.1111111111111111, 0.0, 0.0, 0.0, 0.0, 0.0]
