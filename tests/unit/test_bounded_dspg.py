

def test_count_model_drs_subject_records_in_structured_source() -> None:
    from knowmoredirt import bounded_dspg as b
    from knowmoredirt.query import frame_from_mapping

    source = '[{"name":"Orchid Alpha","status":"ready"},{"name":"Orchid Beta","status":"paused"},{"name":"Orchid Gamma","status":"ready"}]'
    records = {
        "documents": [{"document_id": "d", "rel_path": "objects.raw", "text": source}],
        "chunks": [{"chunk_id": "c", "document_id": "d", "chunk_order": 0}],
        "source_spans": [{"span_id": "s", "document_id": "d", "chunk_id": "c", "char_start": 0, "char_end": len(source), "text": source}],
        "contexts": [{"context_id": "ctx", "kind": "asserted", "parent_context_id": ""}],
        "relations": [{
            "relation_id": "scanner", "relation_type": "record_value", "subject": "schema marker",
            "predicate": "format", "object": "json", "value": "json", "source_span_id": "s", "context_id": "ctx",
            "metadata_json": '{"record_group":"scanner","surface_format":"json_like"}',
        }],
        "drs_conditions": [],
        "drs_condition_arguments": [],
        "mentions": [], "mention_referents": [], "frames": [], "frame_arguments": [], "temporal_edges": [],
        "metadata_records": [], "identity_hypotheses": [], "drs_identity_hypotheses": [], "referents": [], "context_carriers": [],
    }
    for index, (name, status) in enumerate((
        ("Orchid Alpha", "ready"),
        ("Orchid Beta", "paused"),
        ("Orchid Gamma", "ready"),
    )):
        cid = f"dc{index}"
        records["drs_conditions"].append({
            "drs_condition_id": cid, "source_span_id": "s", "context_id": "ctx",
            "predicate": "status", "evidence_surface": f'"status":"{status}"',
        })
        records["drs_condition_arguments"].extend([
            {"drs_condition_id": cid, "role": "subject", "value": name, "evidence_surface": f'"name":"{name}"'},
            {"drs_condition_id": cid, "role": "object", "value": status, "evidence_surface": f'"status":"{status}"'},
        ])
    b._finalize_records(records)
    question = "How many Orchid records have status ready?"
    frame = frame_from_mapping(question, {
        "answer_type": "count",
        "aggregation": "count",
        "target_anchors": ["Orchid records"],
        "requested_relation": "status",
        "relation_terms": ["ready"],
        "constraints": ["ready"],
        "answer_variables": [],
    }, source="model_query_drs")
    target_terms = b._target_terms(frame, question)
    relation_terms = b._relation_terms(frame, question)
    count, evidence = b._count_matching_record_groups(records, frame, target_terms, relation_terms)
    assert count == 2
    assert evidence


def test_count_model_drs_subject_records_in_json_like_raw_source_without_scanner_tags() -> None:
    from knowmoredirt import bounded_dspg as b
    from knowmoredirt.query import frame_from_mapping

    source = '''group: "Orchid Frame"
records: [
{ name: "Orchid Alpha", owner: "Ila Voss", status: "ready" }
{ name: "Orchid Beta", owner: "Niko Rell", status: "paused" }
{ name: "Orchid Gamma", owner: "Tessa Noll", status: "ready" }
]
'''
    records = {
        "documents": [{"document_id": "d", "rel_path": "objects.raw"}],
        "chunks": [{"chunk_id": "c", "document_id": "d", "chunk_order": 0}],
        "source_spans": [{"span_id": "s", "document_id": "d", "chunk_id": "c", "char_start": 0, "char_end": len(source), "surface": source}],
        "contexts": [{"context_id": "ctx", "kind": "asserted", "parent_context_id": ""}],
        "relations": [],
        "drs_conditions": [],
        "drs_condition_arguments": [],
        "mentions": [], "mention_referents": [], "frames": [], "frame_arguments": [], "temporal_edges": [],
        "metadata_records": [], "identity_hypotheses": [], "drs_identity_hypotheses": [], "referents": [], "context_carriers": [],
    }
    for index, (name, status) in enumerate((("Orchid Alpha", "ready"), ("Orchid Beta", "paused"), ("Orchid Gamma", "ready"))):
        cid = f"dc{index}"
        records["drs_conditions"].append({"drs_condition_id": cid, "source_span_id": "s", "context_id": "ctx", "predicate": "status", "evidence_surface": f'status: "{status}"'})
        records["drs_condition_arguments"].extend([
            {"drs_condition_id": cid, "role": "subject", "value": "", "evidence_surface": name},
            {"drs_condition_id": cid, "role": "value", "value": status, "evidence_surface": status},
        ])
    b._finalize_records(records)
    question = "How many Orchid records have status ready?"
    frame = frame_from_mapping(question, {
        "answer_type": "count", "aggregation": "count", "target_anchors": ["Orchid records"],
        "requested_relation": "status", "relation_terms": ["status", "ready"], "constraints": ["ready"], "answer_variables": [],
    }, source="model_query_drs")
    target_terms = b._target_terms(frame, question)
    relation_terms = b._relation_terms(frame, question)
    count, evidence = b._count_matching_record_groups(records, frame, target_terms, relation_terms)
    assert count == 2
    assert evidence


def test_model_drs_record_count_does_not_borrow_value_from_sibling_record_context() -> None:
    from knowmoredirt import bounded_dspg as b
    from knowmoredirt.query import frame_from_mapping

    source = '''records: [
{ name: "Orchid Alpha", status: "ready" }
{ name: "Orchid Beta", status: "paused" }
{ name: "Orchid Gamma", status: "ready" }
]
summary: "Only ready records are valid."
'''
    records = {
        "documents": [{"document_id": "d", "rel_path": "objects.raw"}],
        "chunks": [{"chunk_id": "c", "document_id": "d", "chunk_order": 0}],
        "source_spans": [{"span_id": "s", "document_id": "d", "chunk_id": "c", "char_start": 0, "char_end": len(source), "surface": source}],
        "contexts": [{"context_id": "ctx", "kind": "asserted", "parent_context_id": ""}],
        "relations": [], "drs_conditions": [], "drs_condition_arguments": [],
        "mentions": [], "mention_referents": [], "frames": [], "frame_arguments": [], "temporal_edges": [],
        "metadata_records": [], "identity_hypotheses": [], "drs_identity_hypotheses": [], "referents": [], "context_carriers": [],
    }
    for index, (name, status) in enumerate((("Orchid Alpha", "ready"), ("Orchid Beta", "paused"), ("Orchid Gamma", "ready"))):
        cid = f"dc{index}"
        records["drs_conditions"].append({"drs_condition_id": cid, "source_span_id": "s", "context_id": "ctx", "predicate": "status", "evidence_surface": f'status: "{status}"'})
        records["drs_condition_arguments"].extend([
            {"drs_condition_id": cid, "role": "subject", "value": "", "evidence_surface": name},
            {"drs_condition_id": cid, "role": "value", "value": status, "evidence_surface": status},
        ])
    b._finalize_records(records)
    question = "How many Orchid records are ready?"
    frame = frame_from_mapping(question, {
        "answer_type": "count", "aggregation": "count", "target_anchors": ["Orchid records"],
        "requested_relation": "ready", "relation_terms": ["ready", "how many", "answer", "argument"],
        "constraints": [], "answer_variables": ["How many"],
    }, source="model_query_drs")
    count, _ = b._count_matching_record_groups(records, frame, b._target_terms(frame, question), b._relation_terms(frame, question))
    assert count == 2


def test_model_query_frame_drops_answer_variable_from_grounded_targets_and_relations() -> None:
    from knowmoredirt.query import frame_from_mapping

    frame = frame_from_mapping(
        "Who should review the OAuth callback repair PR before merge?",
        {
            "answer_type": "person",
            "answer_variables": ["reviewer"],
            "target_anchors": ["reviewer", "OAuth callback repair PR"],
            "requested_relation": "should review before merge",
            "relation_terms": ["reviewer", "should review", "OAuth callback repair PR", "before merge"],
            "constraints": ["before merge"],
        },
        source="model_query_drs",
    )
    assert frame.answer_variables == ("reviewer",)
    assert frame.target_anchors == ("OAuth callback repair PR",)
    assert "reviewer" not in frame.relation_terms
    assert "should review" in frame.relation_terms


def test_model_query_frame_drops_organization_answer_variable_from_target_anchor() -> None:
    from knowmoredirt.query import frame_from_mapping

    frame = frame_from_mapping(
        "Which customer is blocked by the telemetry export delay?",
        {
            "answer_type": "organization",
            "answer_variables": ["customer"],
            "target_anchors": ["customer"],
            "requested_relation": "is blocked by the telemetry export delay",
            "relation_terms": ["customer", "blocked by", "telemetry export delay"],
            "constraints": ["blocked by telemetry export delay"],
        },
        source="model_query_drs",
    )
    assert frame.answer_variables == ("customer",)
    assert frame.target_anchors == ()
    assert "customer" not in frame.relation_terms
    assert "telemetry export delay" in frame.relation_terms
