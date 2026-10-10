"""
graph.py
========
LangGraph agent graph definition and compilation.

Assembles the nodes from nodes.py into a directed state graph
and compiles it with a memory checkpointer for conversation persistence.

PEP 8 | OOP | Single Responsibility
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from rag_agent.agent.nodes import (
    generation_node,
    query_rewrite_node,
    retrieval_node,
    should_retry_retrieval,
)
from rag_agent.agent.state import AgentState


class AgentGraphBuilder:
    """
    Constructs and compiles the LangGraph agent state graph.

    The graph implements the following RAG pipeline:

        START
          |
          v
    query_rewrite
          |
          v
      retrieval
          |
          v
    conditional decision
       /       \
      /         \
 generate       end
    |
    v
 generation
    |
    v
   END

    MemorySaver enables multi-turn conversation by storing
    state separately for each thread_id.
    """

    def __init__(self) -> None:
        """Initialize the graph builder with an in-memory checkpointer."""
        self._checkpointer = MemorySaver()

    def build(self):
        """
        Assemble nodes and edges, then compile the graph.

        Returns
        -------
        CompiledStateGraph
            A compiled LangGraph graph ready to invoke or stream.
        """

        # ---------------------------------------------------------------
        # 1. Create the state graph
        # ---------------------------------------------------------------
        graph = StateGraph(AgentState)

        # ---------------------------------------------------------------
        # 2. Add nodes
        # ---------------------------------------------------------------
        graph.add_node(
            "query_rewrite",
            query_rewrite_node,
        )

        graph.add_node(
            "retrieval",
            retrieval_node,
        )

        graph.add_node(
            "generation",
            generation_node,
        )

        # ---------------------------------------------------------------
        # 3. Add normal edges
        # ---------------------------------------------------------------

        # START -> Query Rewrite
        graph.add_edge(
            START,
            "query_rewrite",
        )

        # Query Rewrite -> Retrieval
        graph.add_edge(
            "query_rewrite",
            "retrieval",
        )

        # ---------------------------------------------------------------
        # 4. Add conditional routing after retrieval
        # ---------------------------------------------------------------
        graph.add_conditional_edges(
            "retrieval",
            should_retry_retrieval,
            {
                "generate": "generation",
                "end": END,
            },
        )

        # ---------------------------------------------------------------
        # 5. Generation -> END
        # ---------------------------------------------------------------
        graph.add_edge(
            "generation",
            END,
        )

        # ---------------------------------------------------------------
        # 6. Compile graph with conversation memory
        # ---------------------------------------------------------------
        compiled_graph = graph.compile(
            checkpointer=self._checkpointer
        )

        return compiled_graph


@lru_cache(maxsize=1)
def get_compiled_graph():
    """
    Return the singleton compiled graph.

    The lru_cache ensures that the graph is built only once
    per Python process.

    Returns
    -------
    CompiledStateGraph
        The compiled LangGraph RAG agent.
    """

    builder = AgentGraphBuilder()

    return builder.build()