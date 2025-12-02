"""
Enhanced Multi-Stage Classification System with Rule-Based Overrides
Stage 1: Sentiment Analysis (positive/neutral/negative)
Stage 2: Crisis Classification (only if negative)
Stage 3: Rule-Based Post-Processing
"""
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel  # type: ignore
import numpy as np
from typing import Dict, Tuple, Optional, List
import re

from llm_verifier import LLMVerifier, LLMDecision
from gpt_llm_classifier import AdvancedLLMEnsemble

class RuleBasedFilter:
    """
    First-stage filter using linguistic rules
    Prevents false positives by catching obvious cases
    """
    
    def __init__(self):
        # Positive indicators (override crisis detection) - Comprehensive
        self.positive_keywords = {
            # Core positive emotions
            'happy', 'excited', 'grateful', 'thankful', 'blessed', 'joyful', 'joy',
            'amazing', 'wonderful', 'fantastic', 'great', 'excellent', 'love', 'loved', 'loving',
            'proud', 'thrilled', 'delighted', 'cheerful', 'optimistic', 'hopeful',
            'content', 'peaceful', 'satisfied', 'appreciate', 'appreciation',
            # Positive experiences
            'enjoyed', 'enjoying', 'enjoy', 'pleased', 'satisfied', 'fulfilled',
            'inspired', 'motivated', 'energized', 'uplifted', 'encouraged',
            # Restaurant/Service positive
            'delicious', 'friendly', 'helpful', 'perfect', 'incredible', 'awesome',
            'brilliant', 'outstanding', 'superb', 'marvelous', 'splendid', 'exceptional',
            'recommend', 'highly recommend', 'strongly recommend', 'best', 'favorite', 'favourite',
            # Community/Support positive
            'supportive', 'supporting', 'support', 'caring', 'kind', 'generous',
            'welcoming', 'inclusive', 'helpful', 'understanding', 'compassionate',
            # Achievement positive
            'achieved', 'accomplished', 'succeeded', 'success', 'successful',
            'progress', 'improvement', 'growth', 'development', 'advancement',
            # Confident/Empowered/Motivated keywords (CRITICAL - prevent false positives)
            'confident', 'empowered', 'motivated', 'unstoppable', 'unstoppable',
            'strong', 'powerful', 'capable', 'determined', 'resilient', 'resilience',
            'fearless', 'brave', 'courageous', 'bold', 'fierce', 'mighty',
            'victorious', 'triumphant', 'conquer', 'conquering', 'overcome', 'overcoming',
            'rise', 'rising', 'soar', 'soaring', 'ascend', 'ascending', 'climb', 'climbing',
            'thrive', 'thriving', 'flourish', 'flourishing', 'prosper', 'prospering',
            'unbreakable', 'indomitable', 'invincible', 'formidable', 'dominant',
            'champion', 'warrior', 'fighter', 'survivor', 'winner', 'victor',
            # Positive relationship/love keywords (CRITICAL - prevent false positives on "I will marry her")
            'marry', 'marriage', 'wedding', 'propose', 'proposal', 'engaged', 'engagement',
            'spouse', 'husband', 'wife', 'partner', 'fiancé', 'fiancée', 'bride', 'groom',
            'commitment', 'commit', 'devoted', 'devotion', 'dedicated', 'dedication',
            'cherish', 'cherishing', 'adore', 'adoring', 'treasure', 'treasuring',
            'protect', 'protecting', 'care for', 'caring for', 'support', 'supporting',
            'together', 'forever', 'always', 'soulmate', 'soul mate', 'better half'
        }
        
        # Strong negative indicators
        self.negative_keywords = {
            'hurt', 'pain', 'die', 'kill', 'suicide', 'end', 'hopeless',
            'worthless', 'hate', 'can\'t', 'never', 'nothing', 'empty',
            'alone', 'scared', 'fear', 'threat', 'danger', 'harm',
            'disappointed', 'sad', 'upset', 'miserable', 'depressed',
            'broken', 'failure', 'failed', 'ruined', 'regret',
            # Service complaints
            'terrible', 'awful', 'horrible', 'bad', 'poor', 'disappointing',
            'rude', 'unprofessional', 'frustrated', 'frustrating', 'annoyed',
            'dissatisfied', 'unacceptable', 'worst',
            # Frustration and annoyance (should be stress/emotional_distress, NOT self_harm)
            'frustrating', 'frustrated', 'annoying', 'annoyed', 'irritating', 'irritated',
            'can\'t stand', 'can\'t take', 'driving me crazy', 'making me crazy',
            # Note: "sick" and "sick of" are NOT in negative_keywords to prevent false self-harm triggers
            # They are handled separately in frustration patterns
        }
        
        # Hostile/aggressive/anger indicators (should be stress/emotional_distress, NOT neutral)
        self.hostile_keywords = {
            # Direct hostile commands
            'get lost', 'go away', 'shut up', 'leave me alone', 'fuck off', 'piss off',
            'screw you', 'bugger off', 'buzz off', 'get out', 'get away',
            # Insults and derogatory terms
            'piece of', 'idiot', 'stupid', 'moron', 'jerk', 'asshole', 'bastard',
            'dumb', 'fool', 'loser', 'pathetic', 'worthless', 'scum', 'trash',
            # Hostile expressions
            'hate you', 'can\'t stand', 'disgusting', 'revolting', 'awful person',
            'damn', 'hell', 'crap', 'sucks', 'terrible person', 'horrible',
            # Aggressive language
            'shut your mouth', 'keep quiet', 'stop talking', 'be quiet',
            'you\'re wrong', 'you\'re terrible', 'you suck', 'you\'re awful'
        }
        
        # Hostile/aggressive patterns (comprehensive)
        self.hostile_patterns = [
            # Direct hostile commands
            r'\b(get lost|go away|shut up|leave me alone|fuck off|piss off|screw you|bugger off|buzz off|get out|get away)\b',
            # Insults with "piece of" or similar
            r'\b(piece of|you\'re a|you are a|what a)\s+(shit|jerk|idiot|moron|asshole|loser|bastard|fool|dumb|scum|trash)\b',
            # Hate/disgust expressions (but NOT "sick of" or "fed up with" - those are frustration, not hostility)
            r'\b(i )?(hate|can\'t stand|disgusted by) (you|this|that|it|your)\b',
            # Direct insults
            r'\b(you\'re|you are|you) (so |really |such a )?(stupid|idiot|moron|jerk|pathetic|worthless|dumb|fool|loser|terrible|awful|horrible)\b',
            # Aggressive commands
            r'\b(shut (your )?(mouth|face|up)|keep quiet|stop talking|be quiet|just stop)\b',
            # Negative personal attacks
            r'\b(you (suck|stink|are terrible|are awful|are horrible|are pathetic))\b',
            # Swear words in hostile context
            r'\b(damn|hell|crap|shit) (you|this|that|it)\b',
        ]
        
        # Critical self-harm phrases (must detect) - ONLY threats to SELF
        self.self_harm_patterns = [
            # Direct statements
            r'\b(want to|going to|plan to)\s+(die|kill myself|hurt myself|end (my )?life)\b',
            r'\b(suicide|kill myself|end it all|harm myself)\b',
            r'\b(don\'t want to live|can\'t take this anymore|end (my )?pain)\b',
            r'\b(planning to hurt myself|have a plan to (hurt|kill) myself)\b',
            r'\b(pills to overdose|ready to end it|want to (die|kill myself))\b',
            r'\b(i (want|will|am going) to (hurt|kill|harm) myself)\b',
            # Subtle/ambiguous indicators
            r'\b(wonder if|maybe) (anyone|people) (would notice|would care|would miss) (if )?(i was gone|i disappeared|i wasn\'t here)\b',
            r'\b(world|everyone|people) (would be better off|wouldn\'t miss me|wouldn\'t notice) (without me|if i was gone)\b',
            r'\b(not meant to be here|don\'t belong|shouldn\'t be alive)\b',
            r'\b(thoughts about|thinking about|considering) (ending|hurting|killing) (myself|my life|it all)\b',
            r'\b(have everything i need|have a plan|ready to) (end it|do it|go through with it)\b',
            r'\b(tonight|today|this week) (might be|could be|will be) (the night|the day|it)\b',
            r'\b(i\'ve been planning|i have a plan|i know how) (how|what|when) (i would|i will|to)\b',
        ]
        
        # Threats toward OTHERS (NOT self-harm) - should be unsafe_environment
        # IMPORTANT: Only match harmful verbs, NOT positive relationship verbs like "marry", "love", "protect", "care for"
        self.threats_to_others_patterns = [
            r'\b(want to|going to|plan to|will)\s+(hurt|kill|harm|attack|get|assault|beat|hit|stab|shoot)\s+(you|them|him|her|someone|people|others)\b',
            r'\b(i (want|will|am going) to (hurt|kill|harm|attack|get|assault|beat|hit|stab|shoot) (you|them|him|her|someone|people))\b',
            r'\b(threaten(ing)?|threat(s)?)\s+(you|them|him|her|someone|people|others)\b',
            r'\b(going to (hurt|kill|harm|attack|get|assault|beat|hit|stab|shoot) (you|them|him|her|someone|people))\b',
            r'\b(i (hate|want to hurt|will hurt|will kill|will harm|will attack) (you|them|him|her|someone|people))\b',
        ]
        
        # Keep critical_patterns for backward compatibility (self-harm only)
        self.critical_patterns = self.self_harm_patterns
        
        # Neutral/everyday phrases
        self.neutral_patterns = [
            r'\b(went to|going to|need to|have to)\s+(store|work|school|meeting|grocery|shop|buy)\b',
            r'\b(what time|where is|how do i)\b',
            r'\b(my name is|i work|i study|i live)\b',
            r'\b(went to the store|buy (some |some )?groceries|shopping (for|to buy))\b',
            r'\b(going to (the )?(store|shop|grocery|work|meeting|appointment))\b',
            r'\b(scheduled (for|at)|appointment (on|at)|meeting (starts|at))\b',
            # Past tense recovery indicators
            r'\b(used to|i was|i had) (think|thought|feel|felt) (about|that) (hurting|ending|killing) (myself|my life) (but|and) (now|i\'m|i have)\b',
            r'\b(in a (much )?better place|have support|getting help|in therapy|on medication) (now|and)\b',
        ]
        
        # Positive context patterns (comprehensive)
        self.positive_patterns = [
            # Love and appreciation
            r'\b(love (my |having )?life|feel(ing)? (so )?happy|grateful for|thankful for)\b',
            r'\b(love|love how|loved|really love|absolutely love) (this|that|the|how|it|everything)\b',
            r'\b(i )?(love|adore|cherish|treasure|value|appreciate) (you|this|that|it|the|how)\b',
            # Excitement and anticipation
            r'\b(excited (about|for)|can\'t wait|looking forward|thrilled about)\b',
            # Achievement and pride
            r'\b(proud of|achieved|accomplished|succeeded|made progress|improved)\b',
            # Enjoyment and satisfaction
            r'\b(absolutely (loved|love|enjoyed)|really (loved|enjoyed|liked)|thoroughly enjoyed)\b',
            r'\b(was (so |really |absolutely )?(good|great|amazing|wonderful|fantastic|excellent|delicious|friendly|helpful|perfect|incredible))\b',
            # Recommendations
            r'\b((highly |strongly |definitely )?recommend|best (restaurant|service|experience|place|ever|i\'ve|we\'ve))\b',
            # General positivity
            r'\b(going (so |really )?well|everything is (going )?(great|well|perfect|amazing|wonderful))\b',
            # Supportive communities and relationships
            r'\b(supportive|helpful|kind|caring|understanding|welcoming) (community|people|group|team|family|friends)\b',
            r'\b(community|people|everyone|group|team) (is|are) (so |really |very )?(supportive|helpful|amazing|great|wonderful|kind|caring)\b',
            r'\b(how )?(supportive|helpful|amazing|great|wonderful|kind) (this|that|the) (community|people|group|team) (is|are)\b',
            # Complex positive patterns
            r'\b(despite|even though|although) (all )?(the )?(challenges|difficulties|problems|struggles) (i\'ve faced|i faced|i have), (i\'m|i am) (grateful|thankful|appreciative)\b',
            r'\b(lessons learned|strength (i\'ve )?discovered|growth|progress|improvement|positive change)\b',
            r'\b(learn from|do better|next time|improve|grow from|move forward)\b',
            # Gratitude expressions
            r'\b(so |very |really )?(grateful|thankful|appreciative) (for|that|to have|to be)\b',
            r'\b(thank you|thanks|appreciate it|much appreciated)\b',
            # Positive motivational statements (NEW - to suppress false stress/self-harm)
            r'\b(i )?(will|am going to|gonna) (ace|crush|nail|kill it|rock|excel|succeed|win|dominate) (this|that|the|it|my)?\s*(test|exam|presentation|interview|project|assignment|job|meeting|game|match|competition)\b',
            r'\b(i )?(will|am going to|gonna) (do|be) (great|amazing|excellent|fantastic|outstanding|perfect|awesome|incredible)\b',
            r'\b(i )?(will|am going to|gonna) (make|achieve|get|earn|win) (it|this|that|my goal|success|victory)\b',
            r'\b(i )?(will|am going to|gonna) (show|prove) (them|you|everyone|people|the world)\b',
            r'\b(i )?(will|am going to|gonna) (overcome|beat|conquer|master|handle|tackle) (this|that|it|challenges|obstacles)\b',
            r'\b(i )?(will|am going to|gonna) (succeed|win|excel|thrive|flourish|prosper)\b',
            r'\b(i )?(will|am going to|gonna) (be|become) (better|stronger|smarter|faster|stronger)\b',
            r'\b(i )?(will|am going to|gonna) (give|put) (it|this|that|my) (my )?(all|best|everything|100%)\b',
            # Achievement/Confidence patterns (CRITICAL - "I know I can achieve")
            r'\b(i )?(know|believe|trust|am sure|am certain) (i )?(can|will) (achieve|accomplish|do|succeed|make|get|handle|overcome)\b',
            r'\b(i )?(know|believe|trust) (i )?(can|will) (achieve|accomplish|do|succeed|make|get|handle|overcome) (anything|everything|it|this|that)\b',
            r'\b(i )?(know|believe|trust) (i )?(can|will) (achieve|accomplish|do|succeed|make|get|handle|overcome) (anything|everything) (i )?(put|set) (my )?(mind|effort|focus) (to|on)\b',
            r'\b(i )?(have|got) (the )?(ability|strength|power|skills|capability|confidence) (to|of) (achieve|accomplish|do|succeed|make|get|handle|overcome)\b',
            r'\b(i )?(am|feel) (capable|able|ready|confident|sure|certain) (i )?(can|will) (achieve|accomplish|do|succeed|make|get|handle|overcome)\b',
            r'\b(i )?(am|feel) (capable|able|ready|confident|sure|certain) (of|to) (achieving|accomplishing|doing|succeeding|making|getting|handling|overcoming)\b',
            # Confident/Empowered/Motivated patterns (CRITICAL - prevent false positives on "obstacle", "rise", "fall")
            r'\b(i )?(am|feel|feeling|feel like) (confident|empowered|motivated|unstoppable|strong|powerful|capable|determined|resilient|fearless|brave|courageous|bold|fierce|mighty|unbreakable|indomitable|invincible)\b',
            r'\b(confident|empowered|motivated|unstoppable|strong|powerful|capable|determined|resilient|fearless|brave|courageous|bold|fierce|mighty|unbreakable|indomitable|invincible) (enough|to|that|in|about)\b',
            r'\b(i )?(will|am going to|gonna|can|will) (rise|soar|ascend|climb|thrive|flourish|prosper|overcome|conquer|triumph|succeed|win|excel)\b',
            r'\b(rise|rising|soar|soaring|ascend|ascending|climb|climbing|thrive|thriving|flourish|flourishing|prosper|prospering) (above|beyond|higher|up|to the top|to success|to victory)\b',
            r'\b(overcome|overcoming|conquer|conquering|beat|beating|master|mastering|tackle|tackling|handle|handling) (obstacles|challenges|difficulties|problems|struggles|setbacks|barriers|hurdles)\b',
            r'\b(obstacles|challenges|difficulties|problems|struggles|setbacks|barriers|hurdles) (make|makes|made|will make) (me|us) (stronger|better|smarter|wiser|more resilient|more capable)\b',
            r'\b(i )?(am|feel|feeling) (unstoppable|unbreakable|indomitable|invincible|formidable|dominant|powerful|strong|mighty|fierce)\b',
            r'\b(nothing|no one|nobody|no obstacle|no challenge) (can|will|could) (stop|defeat|break|bring down|hold back) (me|us|you)\b',
            r'\b(i )?(will|am going to|gonna) (rise|rise up|rise higher|soar|climb|ascend) (above|beyond|from|despite|through) (this|that|it|obstacles|challenges|difficulties|struggles)\b',
            r'\b(fall|falling|fell) (down|back) (but|and) (i )?(will|am going to|gonna) (rise|get back up|stand up|bounce back|recover|come back stronger)\b',
            r'\b(every|each|all) (obstacle|challenge|difficulty|problem|struggle|setback|barrier|hurdle) (is|are|will be) (a|an) (stepping stone|opportunity|lesson|chance to grow|way to get stronger)\b',
            r'\b(i )?(am|feel|feeling) (a|an) (champion|warrior|fighter|survivor|winner|victor|hero|leader)\b',
            r'\b(i )?(have|got|gained) (the|this) (power|strength|ability|capability|confidence|determination|resilience) (to|of|for)\b',
            # Positive relationship/love patterns (CRITICAL - prevent false positives on "I will marry her")
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (marry|marriage|wedding|propose|proposal|get engaged|get married) (her|him|you|them|my|our)\b',
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (marry|marriage|wedding|propose|proposal|get engaged|get married)\b',
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (love|cherish|adore|treasure|protect|care for|support|help|be with|spend my life with) (her|him|you|them|my|our)\b',
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (love|cherish|adore|treasure|protect|care for|support|help|be with|spend my life with) (her|him|you|them)\b',
            r'\b(marry|marriage|wedding|propose|proposal|engaged|engagement|spouse|husband|wife|partner|fiancé|fiancée|bride|groom)\b',
            r'\b(commitment|commit|devoted|devotion|dedicated|dedication|together|forever|always|soulmate|soul mate|better half)\b',
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (be|become) (her|his|your|their|my|our) (husband|wife|spouse|partner|fiancé|fiancée)\b',
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (make|ask) (her|him|you|them) (my|our) (wife|husband|spouse|partner|fiancée|fiancé)\b',
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (spend|share) (my|our) (life|future|forever|always) (with|together with) (her|him|you|them)\b',
        ]
    
    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Determine base sentiment using rules
        Returns: ('positive', 'neutral', or 'negative'), confidence
        """
        # Input validation
        if not text or not isinstance(text, str):
            return 'neutral', 0.5
        
        text = text.strip()
        if not text:
            return 'neutral', 0.5
        
        text_lower = text.lower()
        
        # CRITICAL: Check for achievement/confidence patterns FIRST (HIGHEST PRIORITY)
        # This prevents false positives on "I know I can achieve" type statements
        achievement_patterns = [
            r'\b(i )?(know|believe|trust|am sure|am certain) (i )?(can|will) (achieve|accomplish|do|succeed|make|get|handle|overcome)\b',
            r'\b(i )?(know|believe|trust) (i )?(can|will) (achieve|accomplish|do|succeed|make|get|handle|overcome) (anything|everything|it|this|that)\b',
            r'\b(i )?(know|believe|trust) (i )?(can|will) (achieve|accomplish|do|succeed|make|get|handle|overcome) (anything|everything) (i )?(put|set) (my )?(mind|effort|focus) (to|on)\b',
            r'\b(i )?(have|got) (the )?(ability|strength|power|skills|capability|confidence) (to|of) (achieve|accomplish|do|succeed|make|get|handle|overcome)\b',
            r'\b(i )?(am|feel) (capable|able|ready|confident|sure|certain) (i )?(can|will) (achieve|accomplish|do|succeed|make|get|handle|overcome)\b',
        ]
        has_achievement_pattern = any(re.search(pattern, text_lower) for pattern in achievement_patterns)
        achievement_keywords = ['achieve', 'accomplish', 'succeed', 'capable', 'ability', 'confidence', 
                               'know I can', 'believe I can', 'trust I can', 'put my mind to']
        has_achievement_keyword = any(keyword in text_lower for keyword in achievement_keywords)
        
        if has_achievement_pattern or has_achievement_keyword:
            # Strong achievement/confidence signal - classify as positive
            # Check that it's not actually negative
            has_negative_context = any(word in text_lower for word in ['hate', 'hurt', 'pain', 'die', 'kill', 'suicide', 'hopeless'])
            if not has_negative_context:
                return 'positive', 0.98  # Very high confidence for achievement statements
        
        # CRITICAL: Check for positive relationship/love patterns
        # This prevents false positives on "I will marry her" type statements
        relationship_patterns = [
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (marry|marriage|wedding|propose|proposal|get engaged|get married) (her|him|you|them|my|our)\b',
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (love|cherish|adore|treasure|protect|care for|support|help|be with|spend my life with) (her|him|you|them|my|our)\b',
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (be|become) (her|his|your|their|my|our) (husband|wife|spouse|partner|fiancé|fiancée)\b',
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (make|ask) (her|him|you|them) (my|our) (wife|husband|spouse|partner|fiancée|fiancé)\b',
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (spend|share) (my|our) (life|future|forever|always) (with|together with) (her|him|you|them)\b',
        ]
        relationship_keywords = ['marry', 'marriage', 'wedding', 'propose', 'proposal', 'engaged', 'engagement',
                                'spouse', 'husband', 'wife', 'partner', 'fiancé', 'fiancée', 'bride', 'groom',
                                'commitment', 'commit', 'devoted', 'devotion', 'dedicated', 'dedication',
                                'cherish', 'cherishing', 'adore', 'adoring', 'treasure', 'treasuring',
                                'protect', 'protecting', 'care for', 'caring for', 'support', 'supporting',
                                'together', 'forever', 'always', 'soulmate', 'soul mate', 'better half']
        has_relationship_pattern = any(re.search(pattern, text_lower) for pattern in relationship_patterns)
        has_relationship_keyword = any(keyword in text_lower for keyword in relationship_keywords)
        
        if has_relationship_pattern or has_relationship_keyword:
            # Strong positive relationship signal - classify as positive
            # Check that it's not actually negative (sarcasm or coercion)
            has_negative_context = any(word in text_lower for word in ['force', 'make', 'coerce', 'threaten', 'hurt', 'kill', 'harm', 'hate'])
            if not has_negative_context:
                return 'positive', 0.98  # Very high confidence for positive relationship statements
        
        # CRITICAL: Check for confident/empowered/motivated patterns
        # This prevents false positives on words like "obstacle", "rise", "fall" in empowering contexts
        confident_patterns = [
            r'\b(i )?(am|feel|feeling|feel like) (confident|empowered|motivated|unstoppable|strong|powerful|capable|determined|resilient|fearless|brave|courageous|bold|fierce|mighty|unbreakable|indomitable|invincible)\b',
            r'\b(confident|empowered|motivated|unstoppable|strong|powerful|capable|determined|resilient|fearless|brave|courageous|bold|fierce|mighty|unbreakable|indomitable|invincible) (enough|to|that|in|about)\b',
            r'\b(overcome|overcoming|conquer|conquering|beat|beating|master|mastering|tackle|tackling|handle|handling) (obstacles|challenges|difficulties|problems|struggles|setbacks|barriers|hurdles)\b',
            r'\b(obstacles|challenges|difficulties|problems|struggles|setbacks|barriers|hurdles) (make|makes|made|will make) (me|us) (stronger|better|smarter|wiser|more resilient|more capable)\b',
            r'\b(i )?(am|feel|feeling) (unstoppable|unbreakable|indomitable|invincible|formidable|dominant|powerful|strong|mighty|fierce)\b',
            r'\b(nothing|no one|nobody|no obstacle|no challenge) (can|will|could) (stop|defeat|break|bring down|hold back) (me|us|you)\b',
            r'\b(i )?(will|am going to|gonna) (rise|rise up|rise higher|soar|climb|ascend) (above|beyond|from|despite|through) (this|that|it|obstacles|challenges|difficulties|struggles)\b',
            r'\b(fall|falling|fell) (down|back) (but|and) (i )?(will|am going to|gonna) (rise|get back up|stand up|bounce back|recover|come back stronger)\b',
            r'\b(every|each|all) (obstacle|challenge|difficulty|problem|struggle|setback|barrier|hurdle) (is|are|will be) (a|an) (stepping stone|opportunity|lesson|chance to grow|way to get stronger)\b',
        ]
        has_confident_pattern = any(re.search(pattern, text_lower) for pattern in confident_patterns)
        has_confident_keywords = any(keyword in text_lower for keyword in ['confident', 'empowered', 'motivated', 'unstoppable', 'unbreakable', 'indomitable', 'invincible', 'champion', 'warrior', 'fighter', 'survivor', 'winner', 'victor'])
        
        if has_confident_pattern or has_confident_keywords:
            # Strong confident/empowered signal - classify as positive
            return 'positive', 0.95  # Very high confidence for empowering statements
        
        # Check for hostile/aggressive language (should be negative, NOT neutral)
        has_hostile = (
            any(keyword in text_lower for keyword in self.hostile_keywords) or
            any(re.search(pattern, text_lower) for pattern in self.hostile_patterns)
        )
        
        if has_hostile:
            return 'negative', 0.95  # Very high confidence for hostility
        
        # Check for threats to OTHERS first (NOT self-harm)
        for pattern in self.threats_to_others_patterns:
            if re.search(pattern, text_lower):
                return 'negative', 1.0  # Definitely negative, but NOT self-harm
        
        # Check for critical self-harm patterns (threats to SELF)
        for pattern in self.self_harm_patterns:
            if re.search(pattern, text_lower):
                return 'negative', 1.0  # Definitely negative
        
        # Count positive vs negative words
        positive_count = sum(1 for word in self.positive_keywords if word in text_lower)
        negative_count = sum(1 for word in self.negative_keywords if word in text_lower)
        
        # Check for positive patterns
        for pattern in self.positive_patterns:
            if re.search(pattern, text_lower):
                positive_count += 2  # Boost positive signal
        
        # Check for neutral patterns
        for pattern in self.neutral_patterns:
            if re.search(pattern, text_lower):
                if negative_count == 0:
                    return 'neutral', 0.9
        
        # Enhanced sarcasm detection (positive words with negative context)
        sarcasm_indicators = [
            'another', 'just what i needed', 'perfect', 'great', 'wonderful',
            'oh great', 'so happy', 'best day ever', 'could just die',
            'lovely', 'fantastic', 'amazing', 'wonderful day'
        ]
        sarcasm_patterns = [
            r'\b(oh )?(great|wonderful|fantastic|perfect|amazing), (another|just what i needed)',
            r'\b(another )?(wonderful|great|fantastic|perfect|amazing) (day|experience|time)',
            r'\b(so )?(happy|excited|thrilled) (i )?(could )?(just )?(die|end it)',
            r'\b(best )?(day|time|experience) (ever|of my life)',
            r'\b(just )?(what i needed|exactly what i wanted)',
        ]
        has_sarcasm = (
            any(indicator in text_lower for indicator in sarcasm_indicators) and negative_count > 0
        ) or any(re.search(pattern, text_lower) for pattern in sarcasm_patterns)
        
        if has_sarcasm:
            return 'negative', 0.9  # Higher confidence for sarcasm
        
        # Determine sentiment with stricter boundaries
        # POSITIVE: Require clear positive signal (at least 2 indicators OR strong pattern)
        if positive_count >= 2 or (positive_count > 0 and any(re.search(pattern, text_lower) for pattern in self.positive_patterns)):
            # Strong positive signal
            if positive_count >= 3:
                confidence = 0.95
            elif positive_count >= 2:
                confidence = 0.85
            else:
                confidence = 0.75
            # Only return positive if negative count is low (not mixed)
            if negative_count == 0:
                return 'positive', confidence
            elif negative_count == 1 and positive_count >= 3:
                # Strong positive can override single negative
                return 'positive', confidence * 0.9
            # If mixed, check if it's actually negative
            elif negative_count >= 2:
                # More negative than positive - classify as negative
                confidence = min(0.95, 0.6 + (negative_count * 0.1))
                return 'negative', confidence
        
        # NEGATIVE: Require clear negative signal
        if negative_count >= 1:
            # Check if it's a strong negative pattern
            has_strong_negative = (
                negative_count >= 2 or
                any(word in text_lower for word in ['terrible', 'awful', 'horrible', 'hate', 'hurt', 'pain', 'die', 'kill'])
            )
            if has_strong_negative:
                confidence = min(0.95, 0.7 + (negative_count * 0.1))
                return 'negative', confidence
            elif negative_count >= 1 and positive_count == 0:
                # Single negative with no positive - still negative
                confidence = 0.65
                return 'negative', confidence
        
        # NEUTRAL: Only when truly no emotional content OR clear neutral activity
        if positive_count == 0 and negative_count == 0:
            # Check for neutral activity words
            neutral_activity_words = ['store', 'grocery', 'shopping', 'buy', 'bought', 
                                     'meeting', 'appointment', 'work', 'going to', 'went to',
                                     'what time', 'where is', 'how do', 'my name is', 'i work', 'i study']
            if any(word in text_lower for word in neutral_activity_words):
                return 'neutral', 0.85
            # Check for informational/question patterns
            question_patterns = [
                r'^(what|where|when|how|why|who|which|can you|do you|is there)',
                r'\?$',  # Ends with question mark
            ]
            if any(re.search(pattern, text_lower) for pattern in question_patterns):
                return 'neutral', 0.8
            # Default neutral for truly ambiguous content
            return 'neutral', 0.6
        else:
            # Mixed signals - determine based on stronger signal
            if positive_count > 0 and negative_count == 0:
                # Only positive, but weak - still positive
                return 'positive', 0.7
            elif negative_count > 0 and positive_count == 0:
                # Only negative, but weak - still negative
                return 'negative', 0.65
            else:
                # Truly mixed - lean toward negative if any negative present
                if negative_count > 0:
                    return 'negative', 0.6
                else:
                    return 'neutral', 0.6
    
    def check_override(self, text: str, model_predictions: Dict[str, float]) -> Optional[Dict]:
        """
        Check if rules should override model predictions
        Returns override dict if needed, None otherwise
        """
        # Input validation
        if not text or not isinstance(text, str):
            return None
        
        text = text.strip()
        if not text:
            return None
        
        if not isinstance(model_predictions, dict):
            return None
        
        sentiment, confidence = self.analyze_sentiment(text)
        text_lower = text.lower()
        
        # Override -2: Achievement/Confidence statements (HIGHEST PRIORITY)
        # This prevents false positives on "I know I can achieve" type statements
        achievement_patterns_check = [
            r'\b(i )?(know|believe|trust|am sure|am certain) (i )?(can|will) (achieve|accomplish|do|succeed|make|get|handle|overcome)\b',
            r'\b(i )?(know|believe|trust) (i )?(can|will) (achieve|accomplish|do|succeed|make|get|handle|overcome) (anything|everything|it|this|that)\b',
            r'\b(i )?(know|believe|trust) (i )?(can|will) (achieve|accomplish|do|succeed|make|get|handle|overcome) (anything|everything) (i )?(put|set) (my )?(mind|effort|focus) (to|on)\b',
            r'\b(i )?(have|got) (the )?(ability|strength|power|skills|capability|confidence) (to|of) (achieve|accomplish|do|succeed|make|get|handle|overcome)\b',
            r'\b(i )?(am|feel) (capable|able|ready|confident|sure|certain) (i )?(can|will) (achieve|accomplish|do|succeed|make|get|handle|overcome)\b',
        ]
        achievement_keywords_check = ['achieve', 'accomplish', 'succeed', 'capable', 'ability', 'confidence',
                                      'know I can', 'believe I can', 'trust I can', 'put my mind to']
        has_achievement_pattern = any(re.search(pattern, text_lower) for pattern in achievement_patterns_check)
        has_achievement_keyword = any(keyword in text_lower for keyword in achievement_keywords_check)
        
        if has_achievement_pattern or has_achievement_keyword:
            # Strong achievement/confidence signal - FORCE positive classification and suppress ALL risk predictions
            has_negative_context = any(word in text_lower for word in ['hate', 'hurt', 'pain', 'die', 'kill', 'suicide', 'hopeless'])
            has_self_harm = any(re.search(pattern, text_lower) for pattern in self.self_harm_patterns)
            
            if not has_negative_context and not has_self_harm:
                return {
                    'emotion': 'positive',
                    'sentiment': 'safe',
                    'confidence': 0.98,
                    'override_reason': 'Rule: Achievement/Confidence statement (e.g., "I know I can achieve") - suppressing false risk predictions'
                }
        
        # Override -1: Positive relationship/love statements
        # This prevents false positives on "I will marry her" type statements
        relationship_patterns_check = [
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (marry|marriage|wedding|propose|proposal|get engaged|get married) (her|him|you|them|my|our)\b',
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (love|cherish|adore|treasure|protect|care for|support|help|be with|spend my life with) (her|him|you|them|my|our)\b',
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (be|become) (her|his|your|their|my|our) (husband|wife|spouse|partner|fiancé|fiancée)\b',
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (make|ask) (her|him|you|them) (my|our) (wife|husband|spouse|partner|fiancée|fiancé)\b',
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (spend|share) (my|our) (life|future|forever|always) (with|together with) (her|him|you|them)\b',
        ]
        relationship_keywords_check = ['marry', 'marriage', 'wedding', 'propose', 'proposal', 'engaged', 'engagement',
                                      'spouse', 'husband', 'wife', 'partner', 'fiancé', 'fiancée', 'bride', 'groom',
                                      'commitment', 'commit', 'devoted', 'devotion', 'dedicated', 'dedication',
                                      'cherish', 'cherishing', 'adore', 'adoring', 'treasure', 'treasuring',
                                      'protect', 'protecting', 'care for', 'caring for', 'support', 'supporting',
                                      'together', 'forever', 'always', 'soulmate', 'soul mate', 'better half']
        has_relationship_pattern = any(re.search(pattern, text_lower) for pattern in relationship_patterns_check)
        has_relationship_keyword = any(keyword in text_lower for keyword in relationship_keywords_check)
        
        if has_relationship_pattern or has_relationship_keyword:
            # Strong positive relationship signal - FORCE positive classification and suppress ALL risk predictions
            # Check that it's not actually negative (coercion, threats, or real distress)
            has_negative_context = any(word in text_lower for word in ['force', 'coerce', 'threaten', 'hurt', 'kill', 'harm', 'hate', 'die', 'suicide', 'hopeless'])
            # Check for self-harm patterns (should NOT override if actual self-harm)
            has_self_harm = any(re.search(pattern, text_lower) for pattern in self.self_harm_patterns)
            
            if not has_negative_context and not has_self_harm:
                return {
                    'emotion': 'positive',
                    'sentiment': 'safe',
                    'confidence': 0.98,
                    'override_reason': 'Rule: Positive relationship/love statement (e.g., "I will marry her") - suppressing false risk predictions'
                }
        
        # Override 0: Confident/Empowered/Motivated statements
        # This prevents false positives on words like "obstacle", "rise", "fall" in empowering contexts
        confident_keywords = ['confident', 'empowered', 'motivated', 'unstoppable', 'unbreakable', 
                            'indomitable', 'invincible', 'champion', 'warrior', 'fighter', 
                            'survivor', 'winner', 'victor', 'strong', 'powerful', 'capable', 
                            'determined', 'resilient', 'fearless', 'brave', 'courageous', 
                            'bold', 'fierce', 'mighty', 'formidable', 'dominant']
        confident_patterns_check = [
            r'\b(i )?(am|feel|feeling|feel like) (confident|empowered|motivated|unstoppable|strong|powerful|capable|determined|resilient|fearless|brave|courageous|bold|fierce|mighty|unbreakable|indomitable|invincible)\b',
            r'\b(overcome|overcoming|conquer|conquering|beat|beating|master|mastering|tackle|tackling|handle|handling) (obstacles|challenges|difficulties|problems|struggles|setbacks|barriers|hurdles)\b',
            r'\b(obstacles|challenges|difficulties|problems|struggles|setbacks|barriers|hurdles) (make|makes|made|will make) (me|us) (stronger|better|smarter|wiser|more resilient|more capable)\b',
            r'\b(i )?(am|feel|feeling) (unstoppable|unbreakable|indomitable|invincible|formidable|dominant|powerful|strong|mighty|fierce)\b',
            r'\b(nothing|no one|nobody|no obstacle|no challenge) (can|will|could) (stop|defeat|break|bring down|hold back) (me|us|you)\b',
            r'\b(i )?(will|am going to|gonna) (rise|rise up|rise higher|soar|climb|ascend) (above|beyond|from|despite|through) (this|that|it|obstacles|challenges|difficulties|struggles)\b',
            r'\b(fall|falling|fell) (down|back) (but|and) (i )?(will|am going to|gonna) (rise|get back up|stand up|bounce back|recover|come back stronger)\b',
            r'\b(every|each|all) (obstacle|challenge|difficulty|problem|struggle|setback|barrier|hurdle) (is|are|will be) (a|an) (stepping stone|opportunity|lesson|chance to grow|way to get stronger)\b',
        ]
        has_confident_keyword = any(keyword in text_lower for keyword in confident_keywords)
        has_confident_pattern = any(re.search(pattern, text_lower) for pattern in confident_patterns_check)
        
        if has_confident_keyword or has_confident_pattern:
            # Strong confident/empowered signal - FORCE positive classification and suppress ALL risk predictions
            # Check that it's not actually negative (sarcasm or real distress)
            has_actual_negative = any(word in text_lower for word in ['hate', 'hurt', 'pain', 'die', 'kill', 'suicide', 'hopeless', 'worthless', 'terrible', 'awful'])
            # Check for self-harm patterns (should NOT override if actual self-harm)
            has_self_harm = any(re.search(pattern, text_lower) for pattern in self.self_harm_patterns)
            
            if not has_actual_negative and not has_self_harm:
                return {
                    'emotion': 'positive',
                    'sentiment': 'safe',
                    'confidence': 0.95,
                    'override_reason': 'Rule: Confident/Empowered/Motivated statement - suppressing false risk predictions'
                }
        
        # Override 1: Obvious positive → Zero out all crisis labels
        if sentiment == 'positive' and confidence > 0.65:  # Even lower threshold
            # But double-check for sarcasm/negative context
            has_negative = any(word in text_lower for word in self.negative_keywords)
            # Check for positive restaurant/service keywords
            positive_service_words = ['delicious', 'friendly', 'loved', 'amazing', 'wonderful', 
                                     'fantastic', 'excellent', 'great', 'recommend', 'best',
                                     'incredible', 'awesome', 'outstanding', 'perfect']
            has_positive_service = any(word in text_lower for word in positive_service_words)
            
            # Check for sarcasm patterns
            sarcasm_patterns = [
                r'\b(oh )?(great|wonderful|fantastic|perfect), (another|just what i needed)',
                r'\b(another )?(wonderful|great|fantastic|perfect) (day|experience)',
                r'\b(so )?(happy|excited|thrilled) (i )?(could )?(just )?(die|end it)',
                r'\b(best )?(day|time|experience) (ever|of my life)',
            ]
            has_sarcasm = any(re.search(pattern, text_lower) for pattern in sarcasm_patterns)
            
            # Check for metaphorical distress
            metaphorical_patterns = [
                r'\b(drowning|suffocating|trapped|crushing) (in|under|by)',
                r'\b(no one|nobody) (to|can) (throw|give|offer) (me )?(a )?(lifeline|help|support)',
                r'\b(can\'t breathe|can\'t escape|no way out)',
            ]
            has_metaphorical = any(re.search(pattern, text_lower) for pattern in metaphorical_patterns)
            
            # Check for rhetorical questions showing distress
            rhetorical_patterns = [
                r'\b(what\'s|what is) (the )?(point|use) (of|when|if)',
                r'\b(why )?(do|does|did) (i|everything) (even )?(bother|try|matter)',
                r'\b(why )?(does|did) (everything|nothing) (always|ever) (go wrong|work out)',
            ]
            has_rhetorical = any(re.search(pattern, text_lower) for pattern in rhetorical_patterns)
            
            # Don't override if there's sarcasm, metaphorical distress, or rhetorical questions
            if not has_sarcasm and not has_metaphorical and not has_rhetorical and (not has_negative or (has_positive_service and not has_negative)):
                return {
                    'emotion': 'positive',
                    'sentiment': 'safe',
                    'confidence': confidence,
                    'override_reason': 'Rule: Obvious positive content'
                }
        
        # Override 2: Frustration/annoyance (should be stress/emotional_distress, NOT self_harm or neutral)
        frustration_patterns = [
            r'\b(frustrating|frustrated|annoying|annoyed|irritating|irritated)\b',
            r'\b(can\'t stand|can\'t take|cannot stand|cannot take)\b',  # More flexible - matches anywhere
            r'\b(i )?(can\'t|cannot) stand\b',  # Matches "I can't stand" with anything after
            r'\b(driving|making) (me )?(crazy|mad|insane)\b',
            r'\b(this|it) (is|\'s) (so |really |extremely )?(frustrating|annoying|irritating)\b',
            r'\b(keeps|keep) (crashing|happening|going wrong|failing)\b',
            r'\b(i )?(can\'t|cannot) stand (the way|how|what|this|it|them|people|anything|.*?)\b',  # Matches anything after "stand"
        ]
        has_frustration = any(re.search(pattern, text_lower) for pattern in frustration_patterns)
        
        # Make sure it's NOT actual self-harm language (check for specific self-harm patterns)
        actual_self_harm_patterns = [
            r'\b(want to|going to|plan to)\s+(die|kill myself|hurt myself|end (my )?life)\b',
            r'\b(suicide|kill myself|end it all|harm myself)\b',
            r'\b(don\'t want to live|end (my )?pain)\b',
            r'\b(planning to hurt myself|have a plan to (hurt|kill) myself)\b',
        ]
        has_actual_self_harm = any(re.search(pattern, text_lower) for pattern in actual_self_harm_patterns)
        
        if has_frustration and not has_actual_self_harm:
            # Frustration should be stress/emotional_distress, NOT self_harm or neutral
            return {
                'emotion': 'stress',
                'sentiment': 'concerning',
                'confidence': 0.75,
                'override_reason': 'Rule: Frustration/annoyance detected - classified as stress (not self-harm or neutral)'
            }
        
        # Override 2a: Threats to OTHERS → Force unsafe_environment (NOT self-harm)
        for pattern in self.threats_to_others_patterns:
            if re.search(pattern, text_lower):
                return {
                    'emotion': 'unsafe_environment',
                    'sentiment': 'concerning',
                    'confidence': 0.95,
                    'override_reason': 'Rule: Threats toward others detected (NOT self-harm)'
                }
        
        # Override 2b: Critical self-harm patterns → Check if it's ambiguous (low) or direct (high)
        for pattern in self.self_harm_patterns:
            if re.search(pattern, text_lower):
                # Check if it's ambiguous/ideation (low) vs direct plan (high)
                ambiguous_indicators = [
                    'wonder', 'think', 'thoughts', 'maybe', 'sometimes', 'not saying',
                    'would notice', 'would care', 'would miss', 'would be better'
                ]
                direct_indicators = [
                    'planning', 'have a plan', 'going to', 'will', 'tonight', 'today',
                    'have everything', 'ready to', 'know how', 'know what'
                ]
                
                has_ambiguous = any(ind in text_lower for ind in ambiguous_indicators)
                has_direct = any(ind in text_lower for ind in direct_indicators)
                
                # If ambiguous language, classify as self_harm_low
                if has_ambiguous and not has_direct:
                    return {
                        'emotion': 'self_harm_low',
                        'sentiment': 'concerning',
                        'confidence': 0.75,
                        'override_reason': 'Rule: Ambiguous self-harm ideation detected'
                    }
                else:
                    # Direct plan/intent → high risk
                    return {
                        'emotion': 'self_harm_high',
                        'sentiment': 'high_risk',
                        'confidence': 0.95,
                        'override_reason': 'Rule: Direct self-harm plan detected'
                    }
        
        # Override 3: Pure neutral/informational → Not risky
        if sentiment == 'neutral' and confidence > 0.65:  # Lower threshold
            max_crisis_score = max([
                model_predictions.get('self_harm_high', 0),
                model_predictions.get('self_harm_low', 0),
                model_predictions.get('unsafe_environment', 0),
                model_predictions.get('emotional_distress', 0)
            ])
            # Override if neutral daily activity detected
            neutral_activity_words = ['store', 'groceries', 'shopping', 'buy', 'bought', 
                                     'meeting', 'appointment', 'scheduled', 'work', 'going to',
                                     'went to', 'need to', 'have to', 'planning to']
            has_neutral_activity = any(word in text_lower for word in neutral_activity_words)
            
            # Check for informational/question patterns
            question_patterns = [
                r'^(what|where|when|how|why|who|which|can you|do you)',
                r'\?$',  # Ends with question mark
            ]
            is_question = any(re.search(pattern, text_lower) for pattern in question_patterns)
            
            # Override if:
            # 1. Low crisis scores, OR
            # 2. Clear neutral activity with moderate crisis scores, OR
            # 3. Question/informational with low crisis scores
            if (max_crisis_score < 0.5 or 
                (has_neutral_activity and max_crisis_score < 0.65) or
                (is_question and max_crisis_score < 0.55)):
                return {
                    'emotion': 'neutral',
                    'sentiment': 'safe',
                    'confidence': confidence,
                    'override_reason': 'Rule: Neutral everyday language'
                }
        
        return None  # No override needed


class MultiStageClassifier:
    """
    Complete classification system with multiple validation stages
    """
    
    def __init__(self, model_path: str = 'checkpoints/best_calibrated_model_temp.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.rule_filter = RuleBasedFilter()
        self.llm_verifier = LLMVerifier()
        self.llm_ensemble = AdvancedLLMEnsemble()  # GPT/LLM ensemble
        
        # Category-specific thresholds - STRICT to prevent false positives
        # These are MINIMUM thresholds - model will use optimal thresholds from training if available
        self.thresholds = {
            'self_harm_high': 0.85,  # VERY HIGH bar (85%) - only clear suicidal intent (increased from 0.80)
            'self_harm_low': 0.75,   # HIGH bar (75%) - prevent false positives on ideation (increased from 0.70)
            'unsafe_environment': 0.75,  # HIGH bar (75%) - prevent false positives (increased from 0.70)
            'emotional_distress': 0.60,  # Medium-high bar (60%) - balanced (increased from 0.55)
            'stress': 0.55,  # Medium bar (55%) - avoid triggering on positive statements (increased from 0.50)
            'neutral': 0.45  # Lower bar (45%) - allow neutral detection (increased from 0.40)
        }
        
        self.label_names = [
            'neutral', 'stress', 'unsafe_environment',
            'emotional_distress', 'self_harm_low', 'self_harm_high'
        ]
        
        # Load model if exists
        try:
            self.model = self._load_model(model_path)
            print(f"[MODEL] Loaded model from {model_path}")
        except Exception as e:
            print(f"[MODEL] Could not load model: {e}")
            self.model = None
    
    def _load_model(self, model_path):
        """Load trained model with temperature scaling"""
        from bert_classifier import BERTMentalHealthClassifier, TemperatureScaling
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        model = BERTMentalHealthClassifier(n_classes=6, dropout=0.3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        # Load temperature scaling if available
        temp_scaling = TemperatureScaling()
        if 'temperature_params' in checkpoint:
            temp_scaling.load_state_dict(checkpoint['temperature_params'])
        
        return {'model': model, 'temp_scaling': temp_scaling}
    
    def classify(self, text: str) -> Dict:
        """
        Complete multi-stage classification with all safeguards
        """
        # Input validation
        if not text or not isinstance(text, str):
            return {
                'text': str(text) if text else '',
                'emotion': 'neutral',
                'sentiment': 'safe',
                'all_scores': {label: 0.0 for label in self.label_names},
                'predictions': [],
                'error': 'Invalid input: text must be a non-empty string'
            }
        
        text = text.strip()
        if not text:
            return {
                'text': text,
                'emotion': 'neutral',
                'sentiment': 'safe',
                'all_scores': {label: 0.0 for label in self.label_names},
                'predictions': [],
                'error': 'Text cannot be empty'
            }
        
        # Stage 1: Rule-based sentiment analysis
        base_sentiment, sent_confidence = self.rule_filter.analyze_sentiment(text)
        
        # Stage 2: Get model predictions (if model loaded)
        if self.model is not None:
            model_scores = self._get_model_predictions(text)
        else:
            # Fallback to rule-based only
            model_scores = {label: 0.0 for label in self.label_names}
            if base_sentiment == 'positive':
                model_scores['neutral'] = 0.9
            elif base_sentiment == 'neutral':
                model_scores['neutral'] = 0.8
            elif base_sentiment == 'negative':
                model_scores['emotional_distress'] = 0.6
        
        # Stage 3: Check for rule-based overrides
        override = self.rule_filter.check_override(text, model_scores)
        override_applied = False
        if override:
            override_applied = True
            # Update model_scores to reflect override (for accurate confidence spectrum)
            override_scores = model_scores.copy()
            override_emotion = override['emotion']
            
            # Set override emotion score to override confidence
            if override_emotion in override_scores:
                override_scores[override_emotion] = override['confidence']
            
            # Suppress conflicting scores if needed
            if override_emotion == 'positive':
                # AGGRESSIVELY suppress risk scores for positive content (should be near 0%)
                for risk_label in ['self_harm_high', 'self_harm_low', 'unsafe_environment', 'emotional_distress', 'stress']:
                    if risk_label in override_scores:
                        override_scores[risk_label] = 0.0  # Force to 0% for positive statements
            elif override_emotion == 'unsafe_environment':
                # Suppress self-harm scores for threats to others
                for self_harm_label in ['self_harm_high', 'self_harm_low']:
                    if self_harm_label in override_scores:
                        override_scores[self_harm_label] = min(override_scores[self_harm_label], 0.2)
            
            # Generate analysis details for override
            analysis_details = self._generate_analysis_details(
                text, base_sentiment, sent_confidence, [{
                    'label': override['emotion'],
                    'confidence': override['confidence'],
                    'source': 'rule_override'
                }], override_scores, True, None
            )
            
            return {
                'text': text,
                'emotion': override['emotion'],
                'sentiment': override['sentiment'],
                'all_scores': override_scores,  # Use updated scores
                'predictions': [{
                    'label': override['emotion'],
                    'confidence': override['confidence'],
                    'threshold': self.thresholds.get(override['emotion'], 0.5),
                    'source': 'rule_override'
                }],
                'override_applied': True,
                'override_reason': override['override_reason'],
                'analysis_details': analysis_details
            }
        
        # Stage 4: Check for achievement/confidence statements FIRST (HIGHEST PRIORITY) and suppress ALL risk predictions
        text_lower = text.lower()
        achievement_patterns_main = [
            r'\b(i )?(know|believe|trust|am sure|am certain) (i )?(can|will) (achieve|accomplish|do|succeed|make|get|handle|overcome)\b',
            r'\b(i )?(know|believe|trust) (i )?(can|will) (achieve|accomplish|do|succeed|make|get|handle|overcome) (anything|everything|it|this|that)\b',
            r'\b(i )?(know|believe|trust) (i )?(can|will) (achieve|accomplish|do|succeed|make|get|handle|overcome) (anything|everything) (i )?(put|set) (my )?(mind|effort|focus) (to|on)\b',
            r'\b(i )?(have|got) (the )?(ability|strength|power|skills|capability|confidence) (to|of) (achieve|accomplish|do|succeed|make|get|handle|overcome)\b',
            r'\b(i )?(am|feel) (capable|able|ready|confident|sure|certain) (i )?(can|will) (achieve|accomplish|do|succeed|make|get|handle|overcome)\b',
        ]
        achievement_keywords_main = ['achieve', 'accomplish', 'succeed', 'capable', 'ability', 'confidence',
                                     'know I can', 'believe I can', 'trust I can', 'put my mind to']
        has_achievement_pattern_main = any(re.search(pattern, text_lower) for pattern in achievement_patterns_main)
        has_achievement_keyword_main = any(keyword in text_lower for keyword in achievement_keywords_main)
        
        if has_achievement_pattern_main or has_achievement_keyword_main:
            # Achievement/confidence statement - STRONGLY suppress ALL risk predictions
            has_negative_context = any(word in text_lower for word in ['hate', 'hurt', 'pain', 'die', 'kill', 'suicide', 'hopeless'])
            has_self_harm_main = any(re.search(pattern, text_lower) for pattern in self.rule_filter.self_harm_patterns)
            
            if not has_negative_context and not has_self_harm_main:
                # AGGRESSIVELY suppress ALL risk scores (force to 0% for achievement statements)
                model_scores['stress'] = 0.0
                model_scores['emotional_distress'] = 0.0
                model_scores['self_harm_high'] = 0.0
                model_scores['self_harm_low'] = 0.0
                model_scores['unsafe_environment'] = 0.0
                # Boost positive/neutral for achievement statements
                model_scores['neutral'] = max(model_scores.get('neutral', 0), 0.95)
        
        # Stage 4a: Check for positive relationship/love statements and suppress ALL risk predictions
        relationship_patterns_main = [
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (marry|marriage|wedding|propose|proposal|get engaged|get married) (her|him|you|them|my|our)\b',
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (love|cherish|adore|treasure|protect|care for|support|help|be with|spend my life with) (her|him|you|them|my|our)\b',
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (be|become) (her|his|your|their|my|our) (husband|wife|spouse|partner|fiancé|fiancée)\b',
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (make|ask) (her|him|you|them) (my|our) (wife|husband|spouse|partner|fiancée|fiancé)\b',
            r'\b(i )?(will|am going to|gonna|want to|plan to|going to) (spend|share) (my|our) (life|future|forever|always) (with|together with) (her|him|you|them)\b',
        ]
        relationship_keywords_main = ['marry', 'marriage', 'wedding', 'propose', 'proposal', 'engaged', 'engagement',
                                    'spouse', 'husband', 'wife', 'partner', 'fiancé', 'fiancée', 'bride', 'groom',
                                    'commitment', 'commit', 'devoted', 'devotion', 'dedicated', 'dedication',
                                    'cherish', 'cherishing', 'adore', 'adoring', 'treasure', 'treasuring',
                                    'protect', 'protecting', 'care for', 'caring for', 'support', 'supporting',
                                    'together', 'forever', 'always', 'soulmate', 'soul mate', 'better half']
        has_relationship_pattern_main = any(re.search(pattern, text_lower) for pattern in relationship_patterns_main)
        has_relationship_keyword_main = any(keyword in text_lower for keyword in relationship_keywords_main)
        
        if has_relationship_pattern_main or has_relationship_keyword_main:
            # Positive relationship statement - STRONGLY suppress ALL risk predictions
            # Check that it's not actually negative (coercion, threats, or real distress)
            has_negative_context = any(word in text_lower for word in ['force', 'coerce', 'threaten', 'hurt', 'kill', 'harm', 'hate', 'die', 'suicide', 'hopeless'])
            has_self_harm_main = any(re.search(pattern, text_lower) for pattern in self.rule_filter.self_harm_patterns)
            
            if not has_negative_context and not has_self_harm_main:
                # Suppress ALL risk scores aggressively
                model_scores['stress'] = min(model_scores.get('stress', 0), 0.05)  # Very strongly suppress stress
                model_scores['emotional_distress'] = min(model_scores.get('emotional_distress', 0), 0.05)  # Very strongly suppress distress
                model_scores['self_harm_high'] = min(model_scores.get('self_harm_high', 0), 0.01)  # Very strongly suppress self-harm
                model_scores['self_harm_low'] = min(model_scores.get('self_harm_low', 0), 0.01)  # Very strongly suppress self-harm
                model_scores['unsafe_environment'] = min(model_scores.get('unsafe_environment', 0), 0.05)  # Very strongly suppress unsafe
                # Boost positive/neutral for relationship statements
                model_scores['neutral'] = max(model_scores.get('neutral', 0), 0.9)  # Boost neutral
        
        # Stage 4b: Check for confident/empowered/motivated statements and suppress ALL risk predictions
        confident_keywords = ['confident', 'empowered', 'motivated', 'unstoppable', 'unbreakable', 
                            'indomitable', 'invincible', 'champion', 'warrior', 'fighter', 
                            'survivor', 'winner', 'victor', 'strong', 'powerful', 'capable', 
                            'determined', 'resilient', 'fearless', 'brave', 'courageous', 
                            'bold', 'fierce', 'mighty', 'formidable', 'dominant']
        confident_patterns = [
            r'\b(i )?(am|feel|feeling|feel like) (confident|empowered|motivated|unstoppable|strong|powerful|capable|determined|resilient|fearless|brave|courageous|bold|fierce|mighty|unbreakable|indomitable|invincible)\b',
            r'\b(overcome|overcoming|conquer|conquering|beat|beating|master|mastering|tackle|tackling|handle|handling) (obstacles|challenges|difficulties|problems|struggles|setbacks|barriers|hurdles)\b',
            r'\b(obstacles|challenges|difficulties|problems|struggles|setbacks|barriers|hurdles) (make|makes|made|will make) (me|us) (stronger|better|smarter|wiser|more resilient|more capable)\b',
            r'\b(i )?(am|feel|feeling) (unstoppable|unbreakable|indomitable|invincible|formidable|dominant|powerful|strong|mighty|fierce)\b',
            r'\b(nothing|no one|nobody|no obstacle|no challenge) (can|will|could) (stop|defeat|break|bring down|hold back) (me|us|you)\b',
            r'\b(i )?(will|am going to|gonna) (rise|rise up|rise higher|soar|climb|ascend) (above|beyond|from|despite|through) (this|that|it|obstacles|challenges|difficulties|struggles)\b',
            r'\b(fall|falling|fell) (down|back) (but|and) (i )?(will|am going to|gonna) (rise|get back up|stand up|bounce back|recover|come back stronger)\b',
            r'\b(every|each|all) (obstacle|challenge|difficulty|problem|struggle|setback|barrier|hurdle) (is|are|will be) (a|an) (stepping stone|opportunity|lesson|chance to grow|way to get stronger)\b',
        ]
        has_confident_keyword = any(keyword in text_lower for keyword in confident_keywords)
        has_confident_pattern = any(re.search(pattern, text_lower) for pattern in confident_patterns)
        
        if has_confident_keyword or has_confident_pattern:
            # Confident/empowered statement - STRONGLY suppress ALL risk predictions
            # Check that it's not actually negative (sarcasm or real distress)
            has_actual_negative = any(word in text_lower for word in ['hate', 'hurt', 'pain', 'die', 'kill', 'suicide', 'hopeless', 'worthless', 'terrible', 'awful'])
            has_self_harm = any(re.search(pattern, text_lower) for pattern in self.rule_filter.self_harm_patterns)
            
            if not has_actual_negative and not has_self_harm:
                # AGGRESSIVELY suppress ALL risk scores (force to 0% for confident statements)
                model_scores['stress'] = 0.0  # Force to 0%
                model_scores['emotional_distress'] = 0.0  # Force to 0%
                model_scores['self_harm_high'] = 0.0  # Force to 0%
                model_scores['self_harm_low'] = 0.0  # Force to 0%
                model_scores['unsafe_environment'] = 0.0  # Force to 0%
                # Boost positive/neutral for confident statements
                model_scores['neutral'] = max(model_scores.get('neutral', 0), 0.9)  # Boost neutral
        
        # Stage 4c: Check for pure neutral statements and AGGRESSIVELY suppress risk scores
        neutral_activity_patterns = [
            r'\b(meeting|appointment|conference|call) (has been|was|is) (moved|scheduled|rescheduled|at|for)',
            r'\b(meeting|appointment|conference|call) (is|will be|starts) (at|on|in)',
            r'\b(went to|going to|need to|have to) (the )?(store|work|school|gym|library|park)',
            r'\b(what time|where is|when is|how do)',
            r'\b(my name is|i work|i study|i live)',
            r'\b(tomorrow|today|next week|next month) (is|will be) (a )?(public holiday|holiday|weekend|day off)',
            r'\b(public holiday|holiday|weekend|day off)',
            r'\b(the )?(weather|temperature|forecast) (is|will be)',
            r'\b(i )?(finished|completed|did) (my|the) (work|project|assignment|task)',
        ]
        has_neutral_activity = any(re.search(pattern, text_lower) for pattern in neutral_activity_patterns)
        
        if has_neutral_activity and base_sentiment == 'neutral':
            # Pure neutral activity - FORCE all risk scores to 0%
            model_scores['stress'] = 0.0
            model_scores['emotional_distress'] = 0.0
            model_scores['self_harm_high'] = 0.0
            model_scores['self_harm_low'] = 0.0
            model_scores['unsafe_environment'] = 0.0
            # Boost neutral
            model_scores['neutral'] = max(model_scores.get('neutral', 0), 0.9)
        
        # Stage 4b: Check for positive motivational statements and suppress stress/self-harm
        motivational_patterns = [
            r'\b(i )?(will|am going to|gonna) (ace|crush|nail|kill it|rock|excel|succeed|win|dominate)',
            r'\b(i )?(will|am going to|gonna) (do|be) (great|amazing|excellent|fantastic|outstanding)',
            r'\b(i )?(will|am going to|gonna) (make|achieve|get|earn|win) (it|this|that|my goal)',
            r'\b(i )?(will|am going to|gonna) (overcome|beat|conquer|master|handle|tackle)',
            r'\b(i )?(will|am going to|gonna) (succeed|win|excel|thrive|flourish)',
        ]
        has_motivational = any(re.search(pattern, text_lower) for pattern in motivational_patterns)
        
        if has_motivational:
            # Suppress stress, self-harm, and emotional_distress for motivational statements
            model_scores['stress'] = min(model_scores.get('stress', 0), 0.2)  # Suppress stress
            model_scores['emotional_distress'] = min(model_scores.get('emotional_distress', 0), 0.2)  # Suppress distress
            model_scores['self_harm_high'] = min(model_scores.get('self_harm_high', 0), 0.15)  # Suppress self-harm
            model_scores['self_harm_low'] = min(model_scores.get('self_harm_low', 0), 0.15)  # Suppress self-harm
            model_scores['unsafe_environment'] = min(model_scores.get('unsafe_environment', 0), 0.2)  # Suppress unsafe
            # Boost neutral for motivational statements
            model_scores['neutral'] = max(model_scores.get('neutral', 0), 0.7)  # Boost neutral
        
        # Check for hostile/aggressive language (should be stress/emotional_distress, NOT neutral or self-harm)
        has_hostile = (
            any(keyword in text_lower for keyword in self.rule_filter.hostile_keywords) or
            any(re.search(pattern, text_lower) for pattern in self.rule_filter.hostile_patterns)
        )
        
        # Check if it's hostile toward others (not self-harm)
        # Hostile commands like "get lost", "go away", "leave me alone", "you're stupid" are toward OTHERS
        hostile_commands = ['get lost', 'go away', 'shut up', 'leave me alone', 'fuck off', 'piss off']
        has_hostile_command = any(cmd in text_lower for cmd in hostile_commands)
        has_insult_pattern = (
            (any(phrase in text_lower for phrase in ['you\'re', 'you are', 'you']) and 
             any(word in text_lower for word in ['stupid', 'idiot', 'moron', 'jerk', 'terrible', 'awful', 'horrible'])) or
            bool(re.search(r'\b(piece of|you\'re a|you are a)', text_lower))
        )
        is_hostile_toward_others = has_hostile and (has_hostile_command or has_insult_pattern)
        
        if is_hostile_toward_others and base_sentiment == 'negative' and not has_motivational:
            # Hostile language toward others should be stress/emotional_distress (NOT self-harm)
            # FORCE these scores if hostile language detected
            model_scores['stress'] = max(model_scores.get('stress', 0), 0.7)  # Higher minimum
            model_scores['emotional_distress'] = max(model_scores.get('emotional_distress', 0), 0.7)  # Higher minimum
            # FORCE suppress neutral and self-harm if hostile toward others
            model_scores['neutral'] = min(model_scores.get('neutral', 0), 0.2)  # Lower maximum
            # Suppress self-harm if it's hostile toward others (not self)
            model_scores['self_harm_high'] = min(model_scores.get('self_harm_high', 0), 0.3)
            model_scores['self_harm_low'] = min(model_scores.get('self_harm_low', 0), 0.3)
        
        # Check for frustration/annoyance (including "can't stand")
        frustration_words = ['frustrating', 'frustrated', 'annoying', 'annoyed', 'irritating', 'irritated', 
                            "can't stand", "cannot stand", "can't take", "cannot take"]
        has_frustration_words = any(word in text_lower for word in frustration_words)
        # Also check for "can't stand" pattern more flexibly
        has_cant_stand = bool(re.search(r'\b(can\'t|cannot) stand\b', text_lower))
        has_frustration_words = has_frustration_words or has_cant_stand
        
        if has_frustration_words and base_sentiment == 'negative' and not has_motivational:
            # Frustration should be stress/emotional_distress, NOT neutral or self_harm
            model_scores['stress'] = max(model_scores.get('stress', 0), 0.55)  # Boost stress
            model_scores['emotional_distress'] = max(model_scores.get('emotional_distress', 0), 0.55)  # Boost emotional_distress
            # Suppress neutral if frustration detected
            model_scores['neutral'] = min(model_scores.get('neutral', 0), 0.3)  # Lower neutral
            # Suppress self_harm if it's just frustration
            if not any(re.search(pattern, text_lower) for pattern in self.rule_filter.self_harm_patterns):
                model_scores['self_harm_high'] = min(model_scores.get('self_harm_high', 0), 0.3)
                model_scores['self_harm_low'] = min(model_scores.get('self_harm_low', 0), 0.3)
        
        # Stage 5: Apply thresholds and determine final classification
        predictions = []
        for label in self.label_names:
            score = model_scores[label]
            threshold = self.thresholds.get(label, 0.5)
            
            # Special case: Check for threats to others FIRST (before self-harm)
            # This ensures we don't confuse threats to others with self-harm
            if label == 'unsafe_environment':
                # Check if text contains threats to others (not self)
                text_lower_check = text.lower()
                has_threat_to_others = any(
                    re.search(pattern, text_lower_check) 
                    for pattern in self.rule_filter.threats_to_others_patterns
                )
                has_self_harm = any(
                    re.search(pattern, text_lower_check) 
                    for pattern in self.rule_filter.self_harm_patterns
                )
                
                # If threats to others detected, prioritize unsafe_environment
                if has_threat_to_others and not has_self_harm and score >= 0.4:
                    predictions.append({
                        'label': label,
                        'confidence': float(score),
                        'threshold': 0.4,
                        'source': 'model'
                    })
                    continue
                elif score >= threshold:
                    predictions.append({
                        'label': label,
                        'confidence': float(score),
                        'threshold': threshold,
                        'source': 'model'
                    })
                    continue
            
            # Special case: stress requires negative sentiment AND no positive motivational content
            if label == 'stress':
                # Check for positive motivational statements (suppress stress)
                text_lower_check = text.lower()
                motivational_patterns = [
                    r'\b(i )?(will|am going to|gonna) (ace|crush|nail|kill it|rock|excel|succeed|win|dominate)',
                    r'\b(i )?(will|am going to|gonna) (do|be) (great|amazing|excellent|fantastic|outstanding)',
                    r'\b(i )?(will|am going to|gonna) (make|achieve|get|earn|win) (it|this|that|my goal)',
                ]
                has_motivational = any(re.search(pattern, text_lower_check) for pattern in motivational_patterns)
                
                # Suppress stress if positive motivational content detected
                if has_motivational:
                    continue  # Skip stress classification for motivational statements
                
                if base_sentiment == 'negative' and score >= 0.40:  # Slightly higher threshold
                    # Allow stress detection for negative sentiment
                    predictions.append({
                        'label': label,
                        'confidence': float(score),
                        'threshold': 0.40,  # Threshold used
                        'source': 'model'
                    })
                elif base_sentiment != 'negative' and score >= threshold:
                    # Normal threshold for non-negative
                    predictions.append({
                        'label': label,
                        'confidence': float(score),
                        'threshold': threshold,
                        'source': 'model'
                    })
                continue
            
            # Special case: emotional_distress with negative sentiment (lower threshold)
            if label == 'emotional_distress':
                # Check for frustration/annoyance indicators
                frustration_words = ['frustrating', 'frustrated', 'annoying', 'annoyed', 'irritating', 'irritated']
                has_frustration_word = any(word in text_lower for word in frustration_words)
                
                if base_sentiment == 'negative' and score >= 0.3:  # Even lower threshold for negative
                    # Boost score if frustration words present
                    if has_frustration_word:
                        score = max(score, 0.5)  # Boost to at least 50%
                    predictions.append({
                        'label': label,
                        'confidence': float(score),
                        'threshold': 0.3,
                        'source': 'model'
                    })
                    continue
                elif score >= threshold:
                    predictions.append({
                        'label': label,
                        'confidence': float(score),
                        'threshold': threshold,
                        'source': 'model'
                    })
                    continue
                continue
            
            # Normal threshold application
            if score >= threshold:
                predictions.append({
                    'label': label,
                    'confidence': float(score),
                    'threshold': threshold,
                    'source': 'model'
                })
        
        # Sort by confidence
        predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # CRITICAL: Check for confusion between self-harm and threats to others
        text_lower = text.lower()
        has_threat_to_others = any(
            re.search(pattern, text_lower) 
            for pattern in self.rule_filter.threats_to_others_patterns
        )
        has_self_harm = any(
            re.search(pattern, text_lower) 
            for pattern in self.rule_filter.self_harm_patterns
        )
        has_hostile = (
            any(keyword in text_lower for keyword in self.rule_filter.hostile_keywords) or
            any(re.search(pattern, text_lower) for pattern in self.rule_filter.hostile_patterns)
        )
        
        # Check if hostile toward others (not self-harm)
        hostile_commands = ['get lost', 'go away', 'shut up', 'leave me alone', 'fuck off', 'piss off']
        has_hostile_command = any(cmd in text_lower for cmd in hostile_commands)
        has_insult_pattern = (
            (any(phrase in text_lower for phrase in ['you\'re', 'you are', 'you']) and 
             any(word in text_lower for word in ['stupid', 'idiot', 'moron', 'jerk', 'terrible', 'awful', 'horrible'])) or
            bool(re.search(r'\b(piece of|you\'re a|you are a)', text_lower))
        )
        is_hostile_toward_others = has_hostile and (has_hostile_command or has_insult_pattern)
        
        # If hostile language toward others (not self-harm), prioritize stress/emotional_distress
        if is_hostile_toward_others and not has_self_harm:
            # Remove self-harm predictions if it's hostile toward others
            predictions = [p for p in predictions if p['label'] not in ['self_harm_low', 'self_harm_high']]
            # Ensure stress/emotional_distress are in predictions
            if not any(p['label'] == 'stress' for p in predictions):
                stress_score = max(model_scores.get('stress', 0.0), 0.65)
                predictions.append({
                    'label': 'stress',
                    'confidence': float(stress_score),
                    'threshold': 0.5,
                    'source': 'hostile_correction'
                })
            if not any(p['label'] == 'emotional_distress' for p in predictions):
                distress_score = max(model_scores.get('emotional_distress', 0.0), 0.65)
                predictions.append({
                    'label': 'emotional_distress',
                    'confidence': float(distress_score),
                    'threshold': 0.5,
                    'source': 'hostile_correction'
                })
            # Re-sort after adding
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        # If threats to others detected, ensure unsafe_environment is prioritized
        if has_threat_to_others and not has_self_harm:
            # Remove any self-harm predictions (they're incorrect)
            predictions = [p for p in predictions if p['label'] not in ['self_harm_low', 'self_harm_high']]
            # Add unsafe_environment if not already present
            if not any(p['label'] == 'unsafe_environment' for p in predictions):
                unsafe_score = max(model_scores.get('unsafe_environment', 0.0), 0.8)  # Higher confidence
                predictions.insert(0, {
                    'label': 'unsafe_environment',
                    'confidence': float(unsafe_score),
                    'threshold': 0.4,
                    'source': 'rule_correction'
                })
        
        # Determine final emotion and sentiment
        llm_summary = None
        if self.llm_verifier:
            try:
                llm_summary = self.llm_verifier.refine(text)
            except Exception as e:
                print(f"[LLM] Error in LLM verifier: {e}")
                llm_summary = None
        
        # Get GPT/LLM classification if available
        llm_classification = None
        if self.llm_ensemble and hasattr(self.llm_ensemble, 'enabled') and self.llm_ensemble.enabled:
            try:
                llm_classification = self.llm_ensemble.classify(text)
            except Exception as e:
                print(f"[LLM] Error getting LLM classification: {e}")
                llm_classification = None

        if predictions:
            top_label = predictions[0]['label']
            top_confidence = predictions[0]['confidence']
            
            # CRITICAL: Check base_sentiment FIRST to properly differentiate positive/neutral/negative
            # If base_sentiment is clearly positive, override low-confidence risk predictions
            if base_sentiment == 'positive' and sent_confidence >= 0.7:
                # Strong positive sentiment - suppress risk predictions unless very high confidence
                if top_label in ['self_harm_high', 'self_harm_low'] and top_confidence < 0.75:
                    # Low confidence self-harm on positive text - likely false positive
                    emotion = 'positive'
                    sentiment = 'safe'
                elif top_label in ['unsafe_environment', 'emotional_distress', 'stress'] and top_confidence < 0.65:
                    # Low confidence risk on positive text - likely false positive
                    emotion = 'positive'
                    sentiment = 'safe'
                else:
                    # High confidence risk even on positive text - use the prediction
                    emotion = top_label
                    if top_label == 'self_harm_high':
                        sentiment = 'high_risk'
                    elif top_label in ['self_harm_low', 'unsafe_environment', 'emotional_distress']:
                        sentiment = 'concerning'
                    elif top_label == 'stress':
                        sentiment = 'concerning'
                    else:
                        sentiment = 'safe'
            # If base_sentiment is clearly neutral, ensure neutral classification
            elif base_sentiment == 'neutral' and sent_confidence >= 0.7:
                # Strong neutral sentiment - only accept risk predictions if high confidence
                if top_label in ['self_harm_high', 'self_harm_low', 'unsafe_environment'] and top_confidence >= 0.7:
                    # High confidence risk - use it
                    emotion = top_label
                    if top_label == 'self_harm_high':
                        sentiment = 'high_risk'
                    elif top_label in ['self_harm_low', 'unsafe_environment']:
                        sentiment = 'concerning'
                    else:
                        sentiment = 'concerning'
                elif top_label == 'neutral':
                    # Neutral prediction matches neutral sentiment - perfect
                    emotion = 'neutral'
                    sentiment = 'safe'
                elif top_label in ['stress', 'emotional_distress'] and top_confidence < 0.6:
                    # Low confidence stress/distress on neutral text - likely false positive
                    emotion = 'neutral'
                    sentiment = 'safe'
                else:
                    # Use prediction but verify it makes sense
                    emotion = top_label
                    if top_label == 'self_harm_high':
                        sentiment = 'high_risk'
                    elif top_label in ['self_harm_low', 'unsafe_environment', 'emotional_distress']:
                        sentiment = 'concerning'
                    elif top_label == 'stress':
                        sentiment = 'concerning'
                    else:
                        sentiment = 'safe'
            # If base_sentiment is negative, use predictions but verify
            elif base_sentiment == 'negative':
                emotion = top_label
                if top_label == 'self_harm_high':
                    sentiment = 'high_risk'
                elif top_label in ['self_harm_low', 'unsafe_environment', 'emotional_distress']:
                    sentiment = 'concerning'
                elif top_label == 'stress':
                    # Stress from negative sentiment should be concerning, not safe
                    sentiment = 'concerning'
                elif top_label == 'neutral':
                    # Neutral prediction on negative sentiment - likely wrong, use stress
                    emotion = 'stress'
                    sentiment = 'concerning'
                else:
                    sentiment = 'concerning'  # Default to concerning for negative sentiment
            # Fallback for ambiguous sentiment
            else:
                emotion = top_label
                if top_label == 'self_harm_high':
                    sentiment = 'high_risk'
                elif top_label in ['self_harm_low', 'unsafe_environment', 'emotional_distress']:
                    sentiment = 'concerning'
                elif top_label == 'stress':
                    sentiment = 'concerning'
                elif top_label == 'neutral':
                    sentiment = 'safe'
                else:
                    sentiment = 'safe'
        else:
            # No predictions above threshold - use sentiment and scores
            if base_sentiment == 'positive':
                emotion = 'positive'
                sentiment = 'safe'
            elif base_sentiment == 'neutral':
                emotion = 'neutral'
                sentiment = 'safe'
            elif base_sentiment == 'negative':
                # Negative sentiment but no predictions - check scores
                max_stress = model_scores.get('stress', 0)
                max_distress = model_scores.get('emotional_distress', 0)
                
                # If negative sentiment with moderate scores, classify as stress/emotional_distress
                if max_stress >= 0.35 or max_distress >= 0.35:
                    if max_stress >= max_distress:
                        emotion = 'stress'
                        sentiment = 'concerning'
                    else:
                        emotion = 'emotional_distress'
                        sentiment = 'concerning'
                else:
                    emotion = 'stress'  # Default to stress for negative sentiment
                    sentiment = 'concerning'
            else:
                emotion = 'conversational'
                sentiment = 'neutral'

        # Use LLM ensemble classification if available and confident
        if llm_classification and llm_classification.get('confidence', 0) > 0.7:
            emotion = llm_classification['emotion']
            sentiment = llm_classification['sentiment']
            # Override predictions with LLM result
            predictions = [{
                'label': emotion,
                'confidence': llm_classification['confidence'],
                'threshold': 0.7,
                'source': 'llm_ensemble'
            }]
        
        if llm_summary and llm_summary.adjustment:
            if llm_summary.adjustment in ('force_safe', 'reinforce_neutral'):
                sentiment = 'safe'
                if emotion not in ['positive', 'neutral']:
                    emotion = 'neutral'
                if not predictions:
                    predictions.append({
                        'label': 'neutral',
                        'confidence': max(
                            llm_summary.scores.get('safe', 0.0),
                            llm_summary.scores.get('neutral', 0.0)
                        ),
                        'threshold': self.thresholds.get('neutral', 0.45),
                        'source': 'llm_guardrail'
                    })
            elif llm_summary.adjustment == 'suggest_risk':
                # CRITICAL: Only apply risk override if ACTUAL suicidal keywords are present
                # This prevents false positives from anger/toxic language
                text_lower = text.lower()
                suicidal_keywords = [
                    'suicide', 'kill myself', 'hurt myself', 'end my life', 'end it all',
                    'want to die', 'going to die', 'planning to hurt', 'planning to kill',
                    'ready to end', 'have a plan', 'going to hurt myself', 'going to kill myself'
                ]
                has_suicidal_keywords = any(keyword in text_lower for keyword in suicidal_keywords)
                
                # Also check for self-harm patterns
                has_self_harm_pattern = any(
                    re.search(pattern, text_lower) 
                    for pattern in self.rule_filter.self_harm_patterns
                )
                
                # Only force risk if BOTH: high LLM confidence AND actual suicidal keywords
                if has_suicidal_keywords or has_self_harm_pattern:
                    # Verify it's actually high confidence
                    llm_self_harm_score = max(
                        llm_summary.scores.get('self_harm_high', 0.0),
                        llm_summary.scores.get('self_harm_low', 0.0)
                    )
                    if llm_self_harm_score >= 0.75:  # Require 75%+ confidence
                        sentiment = 'high_risk'
                        if not predictions or predictions[0]['label'] != 'self_harm_high':
                            predictions.insert(0, {
                                'label': 'self_harm_high',
                                'confidence': float(llm_self_harm_score),
                                'threshold': self.thresholds.get('self_harm_high', 0.85),
                                'source': 'llm_guardrail_verified'
                            })
                        emotion = 'self_harm_high'
                # If no suicidal keywords, DON'T force risk - let normal thresholds handle it
                # This prevents anger/toxic language from being misclassified as self-harm

        # Ensure all_scores are accurate and match predictions
        # Update all_scores to reflect any corrections made
        final_scores = model_scores.copy()
        
        # If predictions were corrected (e.g., threats to others), update scores
        if predictions:
            for pred in predictions:
                label = pred['label']
                # Ensure the score in all_scores matches the prediction confidence
                if label in final_scores:
                    # Use the prediction confidence if it's more accurate
                    if pred.get('source') == 'rule_correction':
                        final_scores[label] = pred['confidence']
        
        # Ensure all scores are valid floats in [0, 1]
        for label in self.label_names:
            if label not in final_scores:
                final_scores[label] = 0.0
            else:
                score = float(final_scores[label])
                # Clamp to valid range
                final_scores[label] = max(0.0, min(1.0, score))
        
        # Generate comprehensive analysis explanation
        analysis_explanation = self._generate_analysis_explanation(
            text, emotion, sentiment, predictions, base_sentiment, 
            has_hostile, override, llm_summary
        )
        
        return {
            'text': text,
            'emotion': emotion,
            'sentiment': sentiment,
            'all_scores': final_scores,  # Use corrected scores
            'predictions': predictions,
            'override_applied': override is not None,
            'base_sentiment': base_sentiment,
            'base_sentiment_confidence': float(sent_confidence),
            'analysis_explanation': analysis_explanation,  # Comprehensive analysis
            'llm_summary': {
                'scores': llm_summary.scores if llm_summary else None,
                'adjustment': llm_summary.adjustment if llm_summary else None,
                'rationale': llm_summary.rationale if llm_summary else None
            } if llm_summary else None
        }
    
    def _generate_analysis_explanation(self, text: str, emotion: str, sentiment: str,
                                      predictions: List[Dict], base_sentiment: str,
                                      has_hostile: bool, override: Optional[Dict],
                                      llm_summary) -> Dict:
        """Generate comprehensive analysis explanation"""
        text_lower = text.lower()
        explanation = {
            'primary_classification': {
                'emotion': emotion,
                'sentiment': sentiment,
                'confidence': predictions[0]['confidence'] if predictions else 0.0
            },
            'detected_patterns': [],
            'reasoning': [],
            'key_indicators': [],
            'similar_cases': []
        }
        
        # Detect patterns
        if has_hostile:
            explanation['detected_patterns'].append('hostile_language')
            explanation['reasoning'].append('Hostile or aggressive language detected (e.g., insults, commands to leave)')
            explanation['key_indicators'].extend([
                word for word in self.rule_filter.hostile_keywords 
                if word in text_lower
            ][:5])  # Limit to 5
            explanation['similar_cases'].append('Similar to: "Get lost", "Go away", "Shut up", "You\'re an idiot"')
        
        # Check for positive patterns
        if base_sentiment == 'positive':
            positive_patterns_found = []
            for pattern in self.rule_filter.positive_patterns:
                if re.search(pattern, text_lower):
                    if 'love' in pattern:
                        positive_patterns_found.append('love_expression')
                    elif 'supportive' in pattern:
                        positive_patterns_found.append('supportive_community')
                    elif 'grateful' in pattern or 'thankful' in pattern:
                        positive_patterns_found.append('gratitude')
                    elif 'excited' in pattern:
                        positive_patterns_found.append('excitement')
            
            if positive_patterns_found:
                explanation['detected_patterns'].extend(positive_patterns_found)
                explanation['reasoning'].append('Strong positive sentiment detected with clear positive indicators')
                explanation['similar_cases'].append('Similar to: "I love this", "So grateful", "Amazing community", "Thank you"')
        
        # Check for frustration/distress
        frustration_words = ['frustrating', 'frustrated', 'annoying', 'annoyed', 'irritating', 'irritated']
        has_frustration = any(word in text_lower for word in frustration_words)
        if has_frustration and emotion in ['stress', 'emotional_distress']:
            explanation['detected_patterns'].append('frustration')
            explanation['reasoning'].append('Frustration or annoyance expressed - classified as stress/emotional_distress (not self-harm)')
            explanation['similar_cases'].append('Similar to: "This is frustrating", "So annoying", "Can\'t stand this"')
        
        # Check for self-harm
        if emotion in ['self_harm_high', 'self_harm_low']:
            explanation['detected_patterns'].append('self_harm_ideation')
            if emotion == 'self_harm_high':
                explanation['reasoning'].append('Direct self-harm intent or plan detected')
            else:
                explanation['reasoning'].append('Self-harm ideation or thoughts detected (lower risk)')
        
        # Check for threats to others
        if emotion == 'unsafe_environment':
            explanation['detected_patterns'].append('threat_to_others')
            explanation['reasoning'].append('Threats or hostility directed toward others detected')
        
        # Add base sentiment explanation
        explanation['sentiment_analysis'] = {
            'base_sentiment': base_sentiment,
            'interpretation': {
                'positive': 'The text expresses positive emotions, gratitude, or satisfaction',
                'negative': 'The text expresses negative emotions, distress, or dissatisfaction',
                'neutral': 'The text is neutral, informational, or everyday conversation'
            }.get(base_sentiment, 'Sentiment analysis completed')
        }
        
        # Add override explanation if applied
        if override:
            explanation['override_applied'] = True
            explanation['override_reason'] = override.get('override_reason', 'Rule-based override applied')
        
        # Add LLM verification if available
        if llm_summary and llm_summary.rationale:
            explanation['llm_verification'] = {
                'rationale': llm_summary.rationale,
                'adjustment': llm_summary.adjustment
            }
        
        return explanation
    
    def _generate_analysis_details(self, text: str, base_sentiment: str, 
                                   sent_confidence: float, predictions: list,
                                   all_scores: dict, override_applied: bool,
                                   llm_summary) -> dict:
        """Generate detailed analysis explanation for the classification"""
        text_lower = text.lower()
        details = {
            'sentiment_analysis': {
                'detected_sentiment': base_sentiment,
                'confidence': float(sent_confidence),
                'explanation': self._explain_sentiment(text, base_sentiment, sent_confidence)
            },
            'detected_patterns': [],
            'classification_reasoning': [],
            'key_indicators': [],
            'similar_cases': []
        }
        
        # Check for hostile language
        has_hostile = (
            any(keyword in text_lower for keyword in self.rule_filter.hostile_keywords) or
            any(re.search(pattern, text_lower) for pattern in self.rule_filter.hostile_patterns)
        )
        if has_hostile:
            details['detected_patterns'].append({
                'type': 'hostile_aggressive',
                'description': 'Hostile or aggressive language detected',
                'impact': 'Classified as stress/emotional_distress (not neutral)'
            })
            details['key_indicators'].extend([
                kw for kw in self.rule_filter.hostile_keywords if kw in text_lower
            ])
        
        # Check for positive patterns
        has_positive_pattern = any(
            re.search(pattern, text_lower) 
            for pattern in self.rule_filter.positive_patterns
        )
        if has_positive_pattern:
            details['detected_patterns'].append({
                'type': 'positive_signal',
                'description': 'Strong positive language detected',
                'impact': 'Crisis labels suppressed, classified as positive/safe'
            })
        
        # Check for sarcasm
        sarcasm_patterns = [
            r'\b(oh )?(great|wonderful|fantastic|perfect), (another|just what i needed)',
            r'\b(so )?(happy|excited|thrilled) (i )?(could )?(just )?(die|end it)',
        ]
        has_sarcasm = any(re.search(pattern, text_lower) for pattern in sarcasm_patterns)
        if has_sarcasm:
            details['detected_patterns'].append({
                'type': 'sarcasm',
                'description': 'Sarcastic or ironic language detected',
                'impact': 'Classified as negative despite positive words'
            })
        
        # Explain top predictions
        if predictions:
            top_pred = predictions[0]
            details['classification_reasoning'].append({
                'label': top_pred['label'],
                'confidence': float(top_pred['confidence']),
                'source': top_pred.get('source', 'model'),
                'explanation': self._explain_prediction(top_pred['label'], text)
            })
        
        # Explain override if applied
        if override_applied:
            details['classification_reasoning'].append({
                'type': 'rule_override',
                'explanation': 'Rule-based override applied based on detected patterns'
            })
        
        # Similar cases (based on patterns)
        details['similar_cases'] = self._find_similar_cases(text, base_sentiment, predictions)
        
        return details
    
    def _explain_sentiment(self, text: str, sentiment: str, confidence: float) -> str:
        """Explain why this sentiment was detected"""
        text_lower = text.lower()
        
        if sentiment == 'negative':
            if any(kw in text_lower for kw in self.rule_filter.hostile_keywords):
                return f"Negative sentiment detected (confidence: {confidence:.0%}) due to hostile/aggressive language"
            elif any(re.search(pattern, text_lower) for pattern in self.rule_filter.self_harm_patterns):
                return f"Negative sentiment detected (confidence: {confidence:.0%}) due to self-harm indicators"
            else:
                return f"Negative sentiment detected (confidence: {confidence:.0%}) based on negative keywords and context"
        elif sentiment == 'positive':
            if any(re.search(pattern, text_lower) for pattern in self.rule_filter.positive_patterns):
                return f"Positive sentiment detected (confidence: {confidence:.0%}) due to strong positive language patterns"
            else:
                return f"Positive sentiment detected (confidence: {confidence:.0%}) based on positive keywords"
        else:
            return f"Neutral sentiment detected (confidence: {confidence:.0%}) - no strong emotional indicators"
    
    def _explain_prediction(self, label: str, text: str) -> str:
        """Explain why this label was predicted"""
        explanations = {
            'stress': 'Stress detected due to frustration, complaints, or negative experiences',
            'emotional_distress': 'Emotional distress detected due to sadness, anxiety, or overwhelming feelings',
            'self_harm_high': 'High-risk self-harm detected due to direct plans or intent',
            'self_harm_low': 'Low-risk self-harm ideation detected - thoughts but no clear plan',
            'unsafe_environment': 'Unsafe environment detected due to threats toward others',
            'neutral': 'Neutral classification - everyday language with no emotional indicators',
            'positive': 'Positive classification - gratitude, appreciation, or positive experiences'
        }
        return explanations.get(label, f'{label} detected based on model analysis')
    
    def _find_similar_cases(self, text: str, sentiment: str, predictions: list) -> list:
        """Find similar case patterns"""
        text_lower = text.lower()
        similar = []
        
        # Hostile language cases
        if any(kw in text_lower for kw in ['get lost', 'go away', 'shut up', 'piece of']):
            similar.append({
                'pattern': 'Hostile commands or insults',
                'examples': ['Get lost, you piece of *', 'Go away, idiot', 'Shut up, you jerk']
            })
        
        # Positive community cases
        if any(kw in text_lower for kw in ['love', 'supportive', 'community']):
            similar.append({
                'pattern': 'Positive community appreciation',
                'examples': ['I love how supportive this community is', 'This community is amazing', 'Grateful for this supportive group']
            })
        
        # Frustration cases
        if any(kw in text_lower for kw in ['frustrating', 'crashing', 'annoying', 'problem']):
            similar.append({
                'pattern': 'Technical frustration or complaints',
                'examples': ['This app keeps crashing', 'So frustrating when this happens', 'Having problems with this']
            })
        
        return similar
    
    def _get_model_predictions(self, text: str) -> Dict[str, float]:
        """
        Get raw model predictions with accurate confidence scores
        Includes comprehensive error handling and validation
        """
        # Input validation
        if not text or not isinstance(text, str):
            return {label: 0.0 for label in self.label_names}
        
        text = text.strip()
        if not text:
            return {label: 0.0 for label in self.label_names}
        
        # Check if model is loaded
        if self.model is None:
            return {label: 0.0 for label in self.label_names}
        
        try:
            model = self.model['model']
            temp_scaling = self.model['temp_scaling']
        except (KeyError, TypeError):
            return {label: 0.0 for label in self.label_names}
        
        try:
            # Tokenize with error handling
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Get predictions with error handling
            with torch.no_grad():
                try:
                    # Get raw logits from model
                    logits = model(input_ids, attention_mask)
                    
                    # Apply temperature scaling if available and trained
                    if temp_scaling is not None:
                        try:
                            # Check if temperature scaling has been trained (has valid temperature)
                            if hasattr(temp_scaling, 'temperature'):
                                temp_value = temp_scaling.temperature.item() if hasattr(temp_scaling.temperature, 'item') else temp_scaling.temperature
                                if temp_value > 0 and not torch.isnan(temp_scaling.temperature):
                                    logits = temp_scaling(logits)
                        except Exception:
                            # If temperature scaling fails, use raw logits
                            pass
                    
                    # Convert logits to probabilities using sigmoid (for multi-label)
                    probs = torch.sigmoid(logits).cpu().numpy()[0]
                    
                    # Ensure probabilities are in valid range [0, 1]
                    probs = np.clip(probs, 0.0, 1.0)
                    
                    # Ensure no NaN or Inf values
                    probs = np.nan_to_num(probs, nan=0.0, posinf=1.0, neginf=0.0)
                    
                except Exception as e:
                    # If model inference fails, return zero scores
                    print(f"[ERROR] Model inference failed: {e}")
                    return {label: 0.0 for label in self.label_names}
            
            # Return as dict with proper float conversion and validation
            scores = {}
            for label, prob in zip(self.label_names, probs):
                try:
                    # Convert to float and ensure it's a valid number
                    score = float(prob)
                    if np.isnan(score) or np.isinf(score):
                        score = 0.0
                    if not (0.0 <= score <= 1.0):
                        score = max(0.0, min(1.0, score))  # Clamp to [0, 1]
                    scores[label] = score
                except (ValueError, TypeError):
                    scores[label] = 0.0
            
            return scores
            
        except Exception as e:
            # If tokenization or any other step fails, return zero scores
            print(f"[ERROR] Prediction failed: {e}")
            return {label: 0.0 for label in self.label_names}


def initialize_multistage_classifier(model_path: str = 'checkpoints/best_calibrated_model_temp.pt'):
    """Initialize the multi-stage classifier system"""
    classifier = MultiStageClassifier(model_path)
    return classifier


# Test the system
if __name__ == '__main__':
    print("\n" + "="*80)
    print("TESTING MULTI-STAGE CLASSIFICATION SYSTEM")
    print("="*80 + "\n")
    
    classifier = MultiStageClassifier()
    
    test_cases = [
        "I'm feeling happy and excited about my new job!",
        "I love my life and feel grateful for everything!",
        "Today was a normal day, nothing special happened.",
        "I'm worried about my exam tomorrow.",
        "I feel so overwhelmed and sad all the time.",
        "I want to hurt myself and end my pain",
        "I've been thinking about ending it all",
    ]
    
    for text in test_cases:
        result = classifier.classify(text)
        print(f"Text: {text}")
        print(f"→ Emotion: {result['emotion']} | Sentiment: {result['sentiment']}")
        if result.get('override_applied'):
            print(f"→ Override: {result.get('override_reason')}")
        print(f"→ Base Sentiment: {result.get('base_sentiment')} ({result.get('base_sentiment_confidence', 0):.2f})")
        print()
