import numpy as np
from llama_index.core.base_query_engine import BaseQueryEngine
from trulens_eval import Feedback, FeedbackMode, OpenAI, TruLlama
from trulens_eval.feedback import Groundedness


def get_prebuilt_trulens_recorder(engine: BaseQueryEngine, app_id: str) -> TruLlama:
    """Returns a TruLlama instance with prebuilt feedbacks
    for the TruRecorder app.
    """

    provider = OpenAI()

    # Answer relevance - did we answer the question?
    f_qa_relevance = Feedback(
        provider.relevance_with_cot_reasons, name="Answer Relevance"
    ).on_input_output()

    context_selection = TruLlama.select_source_nodes().node.text

    # Context relevance - is the context relevant?
    f_qs_relevance = (
        Feedback(provider.qs_relevance, name="Context Relevance")
        .on_input()
        .on(context_selection)
        .aggregate(np.mean)
    )

    # Groundedness - did we use the context?
    grounded = Groundedness(groundedness_provider=provider)

    f_groundedness = (
        Feedback(grounded.groundedness_measure_with_cot_reasons, name="Groundedness")
        .on(context_selection)
        .on_output()
        .aggregate(grounded.grounded_statements_aggregator)
    )

    return TruLlama(
        engine,
        app_id=app_id,
        feedbacks=[f_qa_relevance, f_qs_relevance, f_groundedness],
    )
