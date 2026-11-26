"""LLM task routing with validated intents and evidence-required scientific answers."""
import json
import re

from app.question_parser import INTENT_CONFIG


def route_question(question, context_compound, judge):
    prompt = (
        'You route questions for ATNP, a natural-product research assistant (not limited to terpenoids). '
        'Classify the ENTIRE request, including follow-ups. Any scientific question or mixed greeting/scientific '
        'request is scientific and requires retrieval. help means questions about using this application, '
        'not how a compound works. Never answer the scientific question yourself. '
        'Return ONLY JSON with domain (in_scope, out_of_scope, uncertain), task '
        '(scientific, greeting, help, out_of_scope), requires_retrieval (boolean), '
        'intents (array of allowed IDs), entity_mention (exact substring from question or null), '
        'reason (short explanation). Do not infer chemical identities or remove suffixes. '
        'Select intents by meaning, not by generic question words: “植物来源是什么” requests only plant_source. '
        'For multiple aspects include all requested intents.\n'
        + json.dumps({'allowed_intents': list(INTENT_CONFIG), 'question': question,
                      'previous_compound': context_compound}, ensure_ascii=False)
    )
    try:
        raw = str(judge(prompt)).strip()
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError('Invalid routing object')
        if result.get('domain') not in {'in_scope', 'out_of_scope', 'uncertain'}:
            raise ValueError('Invalid domain')
        if result.get('task') not in {'scientific', 'greeting', 'help', 'out_of_scope'}:
            raise ValueError('Invalid task')
        if type(result.get('requires_retrieval')) is not bool:
            raise ValueError('Invalid retrieval decision')
        intents = result.get('intents')
        if not isinstance(intents, list) or len(intents) > len(INTENT_CONFIG) or any(
            not isinstance(i, str) or i not in INTENT_CONFIG for i in intents
        ):
            raise ValueError('Invalid intents')
        mention = result.get('entity_mention')
        if mention is not None and (not isinstance(mention, str) or not mention.strip()
                                    or len(mention) > 200 or mention.casefold() not in question.casefold()):
            raise ValueError('Entity mention must occur in the question')
        result['intents'] = list(dict.fromkeys(intents))
        result['model_requires_retrieval'] = result['requires_retrieval']
        result['requires_retrieval'] = result['task'] == 'scientific' or result['requires_retrieval']
        if result['task'] == 'scientific' and not result['intents']:
            raise ValueError('Scientific routing needs at least one intent')
        result['reason'] = str(result.get('reason', ''))[:400]
        result['status'] = 'ok'
        return result
    except Exception as exc:
        return {'status': 'fallback', 'domain': 'uncertain', 'task': 'scientific',
                'requires_retrieval': True, 'intents': [], 'entity_mention': None,
                'error_type': type(exc).__name__}


def apply_routing(parsed, route):
    result = dict(parsed)
    result['task_routing'] = route
    if route['status'] != 'ok' or not route['intents']:
        return result
    groups = [{'intent': intent, 'relations': list(INTENT_CONFIG[intent]['relations']),
               'keywords': list(INTENT_CONFIG[intent]['keywords'])} for intent in route['intents']]
    result.update(intent=route['intents'][0], intents=route['intents'], intent_queries=groups,
                  relations=list(dict.fromkeys(r for g in groups for r in g['relations'])),
                  keywords=list(dict.fromkeys(k for g in groups for k in g['keywords'])))
    return result


def direct_response(route, language):
    messages = {
        'greeting': ('你好！你可以询问天然产物的植物来源、药理作用、靶点或通路。我会根据需要检索 ATNP 证据。',
                     'Hello! Ask about natural-product sources, effects, targets or pathways. I can search ATNP evidence.'),
        'help': ('请输入化合物名称和具体问题，例如“Bergamottin 的植物来源是什么？”。可在输入区选择模型；名称有歧义时先确认候选。回答中的引用编号可打开对应参考文献。',
                 'Enter a compound name and a specific question. Select a model beside the input. Confirm ambiguous names first; use answer citations to inspect references.'),
        'out_of_scope': ('这个问题超出了当前 ATNP 天然产物问答的范围。请提出天然产物及其来源、功能、靶点或机制相关的问题。',
                         'This request is outside the current ATNP natural-product scope. Ask about natural products, sources, functions, targets or mechanisms.')
    }
    return messages[route['task']][0 if language == 'zh' else 1]
