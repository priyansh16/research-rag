def test_parser_extracts_titles(parsed_text):

    assert "[Title] SUMMARY" in parsed_text

    assert "[Title] WORK EXPERIENCE" in parsed_text


def test_parser_extracts_list_items(parsed_text):

    assert "[ListItem]" in parsed_text


def test_parser_extracts_resume_content(parsed_text):

    assert "Gaussian Mixture Models" in parsed_text

    assert "Cisco Systems" in parsed_text


def test_parser_preserves_structure(parsed_text):

    assert "[NarrativeText]" in parsed_text