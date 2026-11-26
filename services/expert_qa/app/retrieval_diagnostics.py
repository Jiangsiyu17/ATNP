"""Report search coverage without turning unselected literature into claims."""
from app.reference_builder import _reference_key


def describe_retrieval(groups, references):
    supported = {_reference_key(r) for r in references}
    lead_map, sources = {}, []
    missing = []
    for group in groups:
        if not group.get('selected_count'):
            missing.append(group['intent'])
        agent = group.get('search_agent') or {}
        fulltext = group.get('fulltext_retrieval')
        if fulltext:
            sources.append({'intent': group['intent'], 'source': 'local_passage', **fulltext})
        if agent:
            if not fulltext:
                sources.append({'intent': group['intent'], 'source': 'local_passage',
                                'status': 'available' if agent.get('passage_index_available') else 'not_configured'})
            external_steps = [s for s in agent.get('trace', []) if s.get('tool') == 'search_external']
            # Decode old saved traces, but do not advertise an external source for new queries.
            if external_steps:
                sources.append({'intent': group['intent'], 'source': 'external_europe_pmc',
                                'search_count': len(external_steps),
                                'returned_count': sum(s.get('returned_count', 0) for s in external_steps),
                                'returned_paper_count': sum(s.get('returned_paper_count', 0) for s in external_steps),
                                'eligible_returned_count': sum(s.get('eligible_returned_count', 0) for s in external_steps),
                                'counts_available': bool(external_steps) and all('returned_count' in s for s in external_steps if s.get('tool_status') == 'ok'),
                                'status': ('ok' if any(s.get('tool_status') == 'ok' for s in external_steps)
                                           else 'error' if external_steps else 'not_searched')})
        for row in agent.get('related_literature', []):
            key = _reference_key(row)
            if key in supported or not row.get('title'):
                continue
            source = row.get('retrieval_source', 'knowledge_graph')
            if key not in lead_map:
                lead_map[key] = {**row, 'intent': group['intent'], 'supports_answer': False,
                                 'retrieval_sources': [source]}
            else:
                lead = lead_map[key]
                if source not in lead['retrieval_sources']:
                    lead['retrieval_sources'].append(source)
                if row.get('source_url'):
                    lead['source_url'] = row['source_url']
    # Merge all provenance before taking the display limit; local rows must not
    # hide a later external hit for the same paper.
    leads = sorted(lead_map.values(), key=lambda row:
                   'external_europe_pmc' not in row['retrieval_sources'])[:5]
    if references:
        status = 'partial' if missing else 'supported'
        message = '部分问题尚无通过筛选的证据。' if missing else None
    else:
        status = 'related_leads_only' if leads else 'no_supporting_evidence'
        message = ('检索到与该化合物相关的文献线索，但现有摘录未通过本问题的证据筛选，不能据此给出结论。'
                   if leads else '本次检索未获得能够支持本问题结论的证据。')
        message += '这不代表该作用不存在，也不代表所有文献中都没有相关研究。'
    return {'evidence_status': status, 'evidence_message': message, 'missing_intents': missing,
            'related_literature': leads, 'source_status': sources}
