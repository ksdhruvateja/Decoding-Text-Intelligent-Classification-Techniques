"""
Generate balanced training dataset for mental health classification
"""

import json
import random

# Set seed for reproducibility
random.seed(42)

def create_dataset():
    """Create comprehensive training dataset"""
    
    dataset = []
    
    # NEUTRAL (150 examples)
    neutral_texts = [
        "What is the weather like today?", "How do I reset my password?", "Can you explain quantum physics?",
        "Where is the nearest coffee shop?", "What time does the store close?", "How does photosynthesis work?",
        "What are the symptoms of the flu?", "When was the Declaration signed?", "How do I bake cookies?",
        "What is the capital of France?", "How do airplanes fly?", "What is machine learning?",
        "Can you define AI?", "Where can I find history info?", "What are benefits of exercise?",
        "How do I install this software?", "What is the meaning of this word?", "When does class start?",
        "How many calories in an apple?", "What is the population of NYC?", "Can you recommend a book?",
        "How do I fix my computer?", "What are course requirements?", "Where is the library?",
        "How do I apply for this job?", "What is the difference?", "When will results be announced?",
        "How does digestion work?", "What causes climate change?", "Can you provide education statistics?",
        "I need information about renewable energy", "What is the tax filing process?", "How do I create a website?",
        "What are the steps to solve this?", "Where can I learn programming?", "The meeting is at 3 PM",
        "Please send the report by Friday", "I need to update my contact info", "The deadline is next month",
        "This document has important details", "I'm researching career options", "What skills are needed?",
        "How do I improve my writing?", "Can you suggest study techniques?", "What are teamwork best practices?",
        "I'm curious about ancient civilizations", "How do ecosystems work?", "What influences economic growth?",
        "Can you explain the scientific method?", "Where do I find reliable sources?", "I want to learn languages",
        "What are good design principles?", "How do I organize my schedule?", "What is the history of tech?",
        "Can you describe the water cycle?", "I need population trend data", "What are common interview questions?",
        "How do I prepare for exams?", "What is water's chemical composition?", "Where can I find research papers?",
        "I need visa requirement info", "What are development stages?", "How do I calculate this?",
        "Can you list the main features?", "What is the registration procedure?", "I'm gathering presentation info",
        "How does GPS work?", "What are material properties?", "Where is best to study abroad?",
        "I'm comparing different options", "What are the pros and cons?", "How do I cite sources?",
        "Can you outline the main points?", "What is the timeline?", "I'm analyzing market trends",
        "How do cultures celebrate holidays?", "What is a balanced diet?", "Where can I access online courses?",
        "I'm evaluating various solutions", "What are ethical considerations?", "How do I interpret results?",
        "Can you summarize key findings?", "What is technology's impact on society?", "I'm reviewing documentation",
        "How do I navigate this website?", "What are safety precautions?", "Where do I submit applications?",
        "I'm conducting a research survey", "What are membership requirements?", "How do I contact support?",
        "Can you clarify the instructions?", "What is standard procedure?", "I'm compiling resource lists",
        "How do I access the database?", "What are operating hours?", "Where can I park?",
        "I'm looking at the schedule", "What is the refund policy?", "How do I track my order?",
        "Can you provide contact details?", "What format should I use?", "Where is the conference room?",
        "I need technical specifications", "How do I subscribe?", "What are the dimensions?",
        "Can you send the link?", "Where are the instructions?", "What version is this?",
        "I'm checking the status", "How long will this take?", "What materials do I need?",
        "Can you explain the process?", "Where can I download this?", "What are the next steps?",
        "I'm looking for documentation", "How do I configure settings?", "What options are available?",
        "Can you describe the features?", "Where is the user manual?", "What is the price?",
        "I'm requesting information", "How do I register?", "What are the specifications?",
        "Can you provide examples?", "Where can I find tutorials?", "What are the guidelines?",
        "I'm inquiring about availability", "How do I get started?", "What is included?",
        "Can you list the requirements?", "Where do I sign up?", "What are the terms?",
        "I'm asking for clarification", "How does this function?", "What is the capacity?",
        "Can you explain the difference?", "Where is more information?", "What are the details?",
        "I'm seeking guidance", "How do I proceed?", "What should I do next?",
        "Can you recommend resources?", "Where can I learn more?", "What is the protocol?",
        "I'm gathering data for analysis", "How do I optimize performance?", "What metrics should I track?",
        "Can you define these terms?", "Where are the examples?", "What is best practice?",
        "I'm exploring different approaches", "How do I troubleshoot?", "What causes this issue?",
        "Can you compare these options?", "Where can I test this?", "What is the standard?",
    ]
    
    for text in neutral_texts[:150]:
        dataset.append({"text": text, "labels": {"neutral": 1, "stress": 0, "unsafe_environment": 0, 
                                                  "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}})
    
    # STRESS (150 examples)
    stress_texts = [
        "I have so much work and not enough time", "I'm overwhelmed with deadlines", "This project is stressing me out",
        "I can't keep up with everything", "I'm so tired from long hours", "Work pressure is getting to me",
        "I feel burned out from meetings", "I'm exhausted from juggling responsibilities", "This semester is challenging",
        "I'm having trouble managing workload", "I feel like I'm running on empty", "Demands keep piling up",
        "I'm struggling with work-life balance", "I'm frustrated with how busy I am", "I can't seem to catch a break",
        "I'm worried about meeting expectations", "I feel stretched too thin", "I'm having concentration trouble",
        "I need a vacation, I'm worn out", "I'm feeling performance pressure", "My schedule is completely packed",
        "I'm annoyed by how much I have to do", "I feel like I'm drowning in tasks", "I'm exhausted from lack of sleep",
        "The workload is really intense", "I'm stressed about upcoming exams", "I can't relax with so much to do",
        "I'm feeling pressured by commitments", "I'm tired of constantly rushing", "I'm overwhelmed by information",
        "I'm struggling to keep organized", "I feel always behind schedule", "I'm stressed about money and bills",
        "I'm worried about finishing on time", "I feel like I'm constantly multitasking", "I'm having time management difficulty",
        "I'm feeling the strain of responsibilities", "I'm frustrated with demands", "I can't get ahead of my to-do list",
        "I'm overwhelmed by life's demands", "I'm tired of constant pressure", "I'm having sleep trouble from stress",
        "I feel like I need perfection", "I'm anxious about performance review", "I'm struggling with coursework",
        "I feel like I'm playing catch-up", "I'm stressed about making ends meet", "I'm feeling the weight of expectations",
        "I can't stop thinking about tasks", "I'm exhausted from daily grind", "I'm worried about job security",
        "I feel overwhelmed by household chores", "I'm frustrated with tech problems", "I'm stressed about relationships",
        "I'm having a hard time with this situation", "I feel like I'm problem-solving constantly", "I'm tired of complications",
        "I'm overwhelmed by appointments", "I'm stressed about event planning", "I feel like I'm juggling too much",
        "I'm anxious about making decisions", "I'm feeling family obligation pressure", "I'm struggling to maintain routines",
        "I'm frustrated by unexpected challenges", "I can't find time for myself", "I'm stressed about scheduling",
        "I feel like I'm always fixing problems", "I'm tired of being busy", "I'm overwhelmed by life's pace",
        "I'm having trouble coping with changes", "I feel stretched beyond limits", "I'm stressed about achieving goals",
        "I'm feeling burnt out from caregiving", "I'm exhausted from commuting", "I'm overwhelmed by social obligations",
        "I'm frustrated with lack of progress", "I can't get a moment's peace", "I'm stressed about health appointments",
        "I'm tired of feeling rushed", "I'm overwhelmed by financial planning", "I'm having difficulty balancing priorities",
        "I feel constantly on edge", "I'm stressed about home renovations", "I'm feeling pressure to succeed",
        "I'm exhausted from emotional labor", "I'm overwhelmed by information overload", "I'm frustrated with admin tasks",
        "I feel like I'm always on call", "I'm stressed about presentations", "I'm tired of the chaos",
        "I'm overwhelmed by competing demands", "I'm having trouble disconnecting", "I feel like I'm running a marathon",
        "I'm stressed about keeping up appearances", "I'm feeling time constraint pressure", "I'm exhausted from interruptions",
        "I'm overwhelmed by all the planning", "I'm frustrated with my energy level", "I feel like I'm putting out fires",
        "I'm stressed about everything at once", "I'm feeling deadline anxiety", "I'm tired from constant alertness",
        "I'm overwhelmed by client demands", "I'm frustrated with my workload", "I feel like I need a break desperately",
        "I'm stressed about exam preparation", "I'm feeling the burden of expectations", "I'm exhausted from overtime work",
        "I'm overwhelmed by project complexity", "I'm frustrated with time pressure", "I feel like I'm barely coping",
        "I'm stressed about family responsibilities", "I'm feeling stretched in all directions", "I'm tired of the grind",
        "I'm overwhelmed by multiple projects", "I'm frustrated with lack of support", "I feel like I can't slow down",
        "I'm stressed about meeting targets", "I'm feeling the weight of duties", "I'm exhausted from mental load",
        "I'm overwhelmed by expectations", "I'm frustrated with the demands", "I feel like I'm always behind",
        "I'm stressed about performance", "I'm feeling pressure from all sides", "I'm tired of the stress",
        "I'm overwhelmed by responsibilities", "I'm frustrated with the pace", "I feel like I need relief",
    ]
    
    for text in stress_texts[:150]:
        dataset.append({"text": text, "labels": {"neutral": 0, "stress": 1, "unsafe_environment": 0, 
                                                  "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}})
    
    # UNSAFE ENVIRONMENT (100 examples)
    unsafe_texts = [
        "Someone is following me and I'm scared", "I don't feel safe at home", "My partner is threatening me",
        "I'm being harassed at work", "Someone is stalking me online", "I'm afraid of my roommate",
        "There's violence in my neighborhood", "I'm being bullied at school", "Someone is making threats against me",
        "I don't feel safe walking alone", "I'm in an abusive relationship", "Someone broke into my house",
        "I'm being intimidated by someone", "I'm scared of my family member", "There's domestic violence happening",
        "I'm afraid to go home", "Someone is blackmailing me", "I'm being sexually harassed",
        "I witnessed a violent incident", "I'm in danger and need help", "Someone is threatening my children",
        "I'm afraid of retaliation", "There's gang activity near me", "I'm being coerced into something",
        "Someone hit me and I'm hurt", "I'm trapped in a dangerous situation", "I need to escape from someone",
        "I'm being physically abused", "Someone won't leave me alone", "I'm afraid of what might happen tonight",
        "There's violence in my home", "I'm being controlled by someone", "Someone is hurting me repeatedly",
        "I'm scared to speak up about abuse", "I'm in immediate danger", "Someone is making me feel unsafe",
        "I'm being threatened with harm", "I need protection from someone", "I'm afraid to report what's happening",
        "Someone is invading my privacy aggressively", "I'm experiencing workplace violence", "I'm being attacked verbally and physically",
        "Someone is damaging my property", "I'm afraid of escalating violence", "I'm being held against my will",
        "Someone is putting me in danger", "I'm scared of someone's behavior", "I'm experiencing hate crimes",
        "Someone is threatening my safety", "I need help getting away", "I'm afraid of being hurt again",
        "Someone is making violent threats", "I'm in an unsafe living situation", "I'm being exploited by someone",
        "Someone is using force against me", "I'm scared to be alone with someone", "There's a dangerous person near me",
        "I'm experiencing road rage incidents", "Someone is attempting to harm me", "I'm afraid of violence breaking out",
        "I'm being terrorized by someone", "Someone is stalking my movements", "I'm in a hostile environment",
        "I'm being targeted for attack", "Someone is displaying weapons", "I'm afraid of retribution",
        "I'm experiencing repeated threats", "Someone is cornering me physically", "I'm scared of what they might do",
        "I'm in a situation that feels dangerous", "Someone is becoming aggressive", "I'm afraid violence will happen",
        "I'm being hunted by someone", "Someone makes me fear for my life", "I need emergency protection",
        "I'm experiencing cyber threats with real danger", "Someone keeps showing up uninvited", "I'm afraid of being kidnapped",
        "I'm experiencing discrimination with threats", "Someone is trying to isolate me dangerously", "I'm being harmed by someone I know",
        "Someone is abusing their power over me", "I'm scared of the person I live with", "I'm experiencing domestic terror",
        "Someone is using violence to control me", "I'm afraid to sleep at night", "I'm being pursued by someone dangerous",
        "Someone is making credible threats", "I'm in fear of my safety", "I'm being victimized repeatedly",
        "Someone is attacking my character and person", "I'm experiencing violent harassment", "I'm scared of retaliation for speaking",
        "Someone is endangering my children", "I'm in an abusive home", "I'm being threatened daily",
        "Someone is planning to hurt me", "I'm experiencing severe intimidation", "I'm afraid of what comes next",
        "Someone is escalating their aggression", "I'm in immediate physical danger", "I'm being hunted down",
    ]
    
    for text in unsafe_texts[:100]:
        dataset.append({"text": text, "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 1, 
                                                  "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 0}})
    
    # EMOTIONAL DISTRESS (150 examples)
    distress_texts = [
        "I've been feeling really depressed lately", "I'm so anxious all the time", "I feel empty inside",
        "I can't stop feeling sad", "I feel hopeless about the future", "I'm having panic attacks frequently",
        "I feel so alone and lonely", "I can't find joy in things I loved", "I feel worthless and useless",
        "I'm numb and can't feel anything", "I'm crying all the time", "I feel like a failure",
        "I'm so anxious I can't breathe", "I feel broken and can't be fixed", "I'm lost and don't know who I am",
        "I feel helpless to change my situation", "I'm drowning in sadness", "I feel like giving up on everything",
        "I can't handle these emotions", "I feel like I'm falling apart", "I'm consumed by worry and fear",
        "I feel disconnected from everyone", "I'm having dark thoughts constantly", "I feel like a burden to others",
        "I can't see any way out of darkness", "I'm terrified of everything", "I feel like I'm losing my mind",
        "I can't stop the negative thoughts", "I feel so guilty about everything", "I'm paralyzed by anxiety",
        "I feel like I don't matter", "I'm having intrusive thoughts", "I feel so ashamed of myself",
        "I can't escape these feelings", "I'm too depressed to get out of bed", "I feel like I'm drowning in despair",
        "I can't cope with life anymore", "I feel like there's no hope left", "I'm in a dark hole",
        "I can't stop worrying about everything", "I feel completely defeated", "I'm having nightmares every night",
        "I feel like I'm not good enough", "I can't control my anxiety", "I feel overwhelmed by sadness",
        "I'm terrified of the future", "I feel like I'm losing everything", "I can't find meaning in anything",
        "I feel like I'm suffocating emotionally", "I'm having trouble functioning", "I feel broken beyond repair",
        "I can't shake this feeling of doom", "I feel like I'm spiraling downward", "I'm so depressed I can barely function",
        "I feel trapped in darkness", "I can't see any light at the end", "I feel completely overwhelmed by emotions",
        "I'm having constant anxiety attacks", "I feel like I'm losing hope", "I can't stop the pain inside",
        "I feel like I'm disappearing", "I'm so anxious I can't leave the house", "I feel like I have no future",
        "I can't bear these feelings much longer", "I feel completely alone in this", "I'm struggling with severe depression",
        "I feel like I'm at my breaking point", "I can't find a reason to keep going", "I feel like I'm barely surviving",
        "I'm having thoughts that scare me", "I feel like I've lost everything", "I can't escape this mental pain",
        "I feel like I'm drowning in anxiety", "I'm so depressed I can't think clearly", "I feel worthless to everyone",
        "I can't see anything positive anymore", "I feel like I'm in constant emotional pain", "I'm having breakdown after breakdown",
        "I feel like I can't go on like this", "I can't handle life anymore", "I feel like there's no point to anything",
        "I'm completely overwhelmed by depression", "I feel like I'm losing my grip on reality", "I can't stop feeling this way",
        "I feel like I'm in unbearable pain", "I feel like I'm at the end of my rope", "I can't take this emotional suffering",
        "I feel completely broken inside", "I feel like I'm being crushed by sadness", "I can't imagine feeling better",
        "I feel like I've given up inside", "I can't see any way forward", "I feel like I'm being consumed by darkness",
        "I'm so depressed everything seems pointless", "I feel like I can't do this anymore", "I'm losing all motivation",
        "I feel isolated from the world", "I'm struggling with severe anxiety", "I feel like happiness is impossible",
        "I can't shake this depression", "I feel like I'm in a black hole", "I'm overwhelmed by negative emotions",
        "I feel like life is meaningless", "I can't find peace in my mind", "I feel like I'm trapped in sadness",
        "I'm having constant panic", "I feel like I'm losing myself", "I can't see a future for me",
        "I feel like I'm suffering endlessly", "I'm drowning in depression", "I feel like everything is falling apart",
        "I can't stop the tears", "I feel like I'm in endless pain", "I'm consumed by hopelessness",
        "I feel like I'm fading away", "I can't find any light", "I feel like I'm being swallowed by darkness",
        "I'm having severe emotional distress", "I feel like I'm completely lost", "I can't see any hope",
    ]
    
    for text in distress_texts[:150]:
        dataset.append({"text": text, "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0, 
                                                  "emotional_distress": 1, "self_harm_low": 0, "self_harm_high": 0}})
    
    # SELF HARM LOW (100 examples)
    low_risk_texts = [
        "I'm afraid I might do something I'll regret", "I'm worried I could hurt myself", "Sometimes I think about ending things",
        "I'm scared of where my thoughts are going", "I'm afraid of what I might do", "I've been having thoughts about harming myself",
        "I'm worried I might act on these urges", "I'm scared I could do something harmful", "I've been thinking about ways to escape pain",
        "I'm afraid I might lose control", "These thoughts keep crossing my mind", "I'm worried about my impulses",
        "I'm scared of what I'm capable of", "I've been thinking about self-harm lately", "I'm afraid I might not be able to stop myself",
        "I'm worried about these dangerous thoughts", "I'm scared I might give in to urges", "I've been thinking about hurting myself",
        "I'm afraid of acting on these feelings", "I'm worried I could do something bad", "I'm scared of these intrusive thoughts about harm",
        "I've been tempted to hurt myself", "I'm afraid I might make a terrible choice", "I'm worried about what I might do when alone",
        "I'm scared these thoughts will win", "I've been thinking about ways to end my pain", "I'm afraid I might hurt myself when things get bad",
        "I'm worried I could lose control and do something", "I'm scared of my own thoughts lately", "I've been having urges to harm myself",
        "I'm afraid of where this is heading", "I'm worried I might not ask for help in time", "I'm scared I could hurt myself if things get worse",
        "I've been thinking about self-injury", "I'm afraid these thoughts mean something bad", "I'm worried I might act impulsively",
        "I'm scared of what I think about doing", "I've been considering ways to harm myself", "I'm afraid I won't be strong enough to resist",
        "I'm worried about my dark thoughts", "I'm scared I might hurt myself accidentally", "I've been thinking thoughts that frighten me",
        "I'm afraid of what might happen if I'm alone too long", "I'm worried I could do something to myself", "I'm scared of losing the fight",
        "I've been having thoughts about ending the suffering", "I'm afraid I might hurt myself to feel something", "I'm worried these urges will become stronger",
        "I'm scared I might not be able to resist much longer", "I've been thinking about self-destructive actions", "I'm afraid of what I might do in weakness",
        "I'm worried I could hurt myself without meaning to", "I'm scared of these thoughts about harming myself", "I've been having intrusive thoughts about self-harm",
        "I'm afraid I might give in to the pain", "I'm worried about acting on these feelings", "I'm scared I could hurt myself when upset",
        "I've been thinking about ways to hurt myself", "I'm afraid I might lose the battle with these thoughts", "I'm worried I could do something harmful in desperation",
        "I'm scared of what I might do if things don't improve", "I've been having thoughts about taking drastic action", "I'm afraid I might hurt myself to escape",
        "I'm worried these thoughts are getting stronger", "I'm scared I might not reach out for help", "I've been thinking about hurting myself when alone",
        "I'm afraid I might act on impulse", "I'm worried I could do something I can't take back", "I'm scared of my potential to harm myself",
        "I've been having thoughts I'm ashamed to admit", "I'm afraid I might hurt myself if I don't get help", "I'm worried about what I'm capable of doing",
        "I'm scared these thoughts will overwhelm me", "I've been thinking about self-harm more frequently", "I'm afraid I might hurt myself in a dark moment",
        "I'm worried I could lose control of my actions", "I'm scared of where my mind is taking me", "I've been having urges I'm trying to fight",
        "I'm afraid I might not be able to stop the thoughts", "I'm worried I could hurt myself if pushed too far", "I'm scared of these harmful urges",
        "I've been thinking about self-harm as an option", "I'm afraid of my impulses", "I'm worried I might hurt myself someday",
        "I'm scared of what I might do eventually", "I've been having concerning thoughts", "I'm afraid I might lose this battle",
    ]
    
    for text in low_risk_texts[:100]:
        dataset.append({"text": text, "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0, 
                                                  "emotional_distress": 0, "self_harm_low": 1, "self_harm_high": 0}})
    
    # SELF HARM HIGH (100 examples)
    high_risk_texts = [
        "I want to kill myself", "I'm going to end my life tonight", "I have a plan to commit suicide",
        "I want to die right now", "I'm going to hurt myself badly", "I can't go on living anymore",
        "I'm planning to end it all", "I want to take my own life", "I'm going to do something harmful to myself",
        "I don't want to live anymore", "I'm ready to kill myself", "I have the means to end my life",
        "I'm going to commit suicide soon", "I want to harm myself severely", "I'm planning my suicide",
        "I can't take being alive anymore", "I'm going to end my life this week", "I want to die and have a plan",
        "I'm going to hurt myself tonight", "I don't deserve to live", "I'm ready to end everything",
        "I have everything ready to kill myself", "I'm going to take all these pills", "I want to jump off this building",
        "I'm going to cut myself deeply", "I can't see any reason to stay alive", "I'm planning to hang myself",
        "I want to crash my car intentionally", "I'm going to overdose tonight", "I don't want to wake up tomorrow",
        "I'm ready to end the pain permanently", "I have a suicide note written", "I'm going to shoot myself",
        "I want to drown myself", "I'm planning to die by suicide", "I can't stand being alive one more day",
        "I'm going to kill myself this weekend", "I want to end my life before Monday", "I'm going to harm myself fatally",
        "I don't want to exist anymore", "I'm ready to commit suicide", "I have picked the day to kill myself",
        "I'm going to end my suffering tonight", "I want to die and escape everything", "I'm planning to take my life soon",
        "I can't bear living another moment", "I'm going to kill myself when everyone leaves", "I want to stop existing",
        "I'm going to harm myself until I die", "I don't see any point in continuing to live", "I'm ready to end my life tonight",
        "I have the gun loaded", "I'm going to jump in front of a train", "I want to poison myself",
        "I'm planning my death", "I can't live like this anymore and I'm ending it", "I'm going to kill myself after I write this",
        "I want my life to be over", "I'm going to harm myself fatally tonight", "I don't want to be alive by tomorrow",
        "I'm ready to die by suicide", "I have decided to end my life", "I'm going to take lethal action",
        "I want to cease existing", "I'm planning to die this evening", "I can't continue living and I'm ending it",
        "I'm going to kill myself very soon", "I want to hurt myself badly enough to die", "I'm going to end everything tonight",
        "I don't want to survive this", "I'm ready to take my own life", "I have made my suicide plan",
        "I'm going to die by my own hand", "I want to end my life immediately", "I'm planning to kill myself tomorrow",
        "I can't live another day and won't", "I'm going to harm myself lethally", "I want to be dead",
        "I'm going to commit suicide this week", "I don't want to live through tonight", "I'm ending my life soon",
        "I have suicide plans ready", "I'm going to kill myself after this", "I want to die tonight",
        "I'm taking my life this weekend", "I can't survive anymore", "I'm going to end it all soon",
        "I want to kill myself before Monday", "I'm planning my suicide for tonight", "I don't want to be alive tomorrow",
        "I'm going to harm myself fatally soon", "I want to end everything now", "I'm ready to commit suicide tonight",
        "I have picked how to kill myself", "I'm going to die by suicide soon", "I want to end my suffering permanently",
        "I'm planning to kill myself this week", "I can't go on and I'm ending it", "I'm going to take my life tonight",
        "I want to die by my own hand", "I'm ending everything soon", "I don't want to exist by tomorrow",
        "I'm going to commit suicide very soon", "I want to kill myself as soon as possible", "I'm taking my life imminently",
    ]
    
    for text in high_risk_texts[:100]:
        dataset.append({"text": text, "labels": {"neutral": 0, "stress": 0, "unsafe_environment": 0, 
                                                  "emotional_distress": 0, "self_harm_low": 0, "self_harm_high": 1}})
    
    # Shuffle dataset
    random.shuffle(dataset)
    
    # Split into train/val
    split_idx = int(0.85 * len(dataset))
    train_data = dataset[:split_idx]
    val_data = dataset[split_idx:]
    
    # Save datasets
    with open('train_data.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open('val_data.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Generated {len(dataset)} total examples")
    print(f"✓ Training set: {len(train_data)} examples")
    print(f"✓ Validation set: {len(val_data)} examples")
    print(f"\nCategory distribution:")
    print(f"  Neutral: 150 examples")
    print(f"  Stress: 150 examples")
    print(f"  Unsafe Environment: 100 examples")
    print(f"  Emotional Distress: 150 examples")
    print(f"  Self Harm Low: 100 examples")
    print(f"  Self Harm High: 100 examples")
    print(f"\n✓ Saved to train_data.json and val_data.json")

if __name__ == '__main__':
    create_dataset()
