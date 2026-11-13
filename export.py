"""Markdown and PowerPoint exports for generated slides."""
from __future__ import annotations
from io import BytesIO
from typing import Any


def slides_to_markdown(slides: list[dict[str, Any]]) -> str:
    parts = ["# Research Paper Presentation\n"]
    for i, slide in enumerate(slides, 1):
        parts.append(f"## Slide {i}: {slide.get('title', 'Untitled')}\n")
        parts.extend(f"- {bullet}" for bullet in slide.get("bullets", []))
        notes = slide.get("speaker_notes", "")
        if notes:
            parts.append(f"\n**Speaker notes:** {notes}")
        parts.append("")
    return "\n".join(parts)


def slides_to_pptx(slides: list[dict[str, Any]]) -> bytes:
    from pptx import Presentation
    from pptx.util import Inches, Pt
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    for item in slides:
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        slide.shapes.title.text = str(item.get("title", "Untitled"))
        title_tf = slide.shapes.title.text_frame
        for p in title_tf.paragraphs:
            p.font.size = Pt(28)
        body = slide.placeholders[1].text_frame
        body.clear()
        for i, bullet in enumerate(item.get("bullets", [])):
            p = body.paragraphs[0] if i == 0 else body.add_paragraph()
            p.text = str(bullet)
            p.level = 0
            p.font.size = Pt(20)
        notes = item.get("speaker_notes", "")
        if notes:
            slide.notes_slide.notes_text_frame.text = str(notes)
    output = BytesIO()
    prs.save(output)
    return output.getvalue()
