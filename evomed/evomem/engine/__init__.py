"""Diagnostic Engine (Engine) reproduction framework."""

from .playbook import (
    Bullet,
    Playbook,
    # Modular experience library
    ModularBullet,
    ModularPlaybook,
    FixedModules,
    MutableModules,
    ContextualStates,
    DecisionBehaviors,
    Uncertainty,
    DelayedAssumptions,
)
from .delta import DeltaOperation, DeltaBatch
from .llm import LLMClient, DummyLLMClient, UniversalLLMClient
from .roles import (
    Reflector,
    Curator,
    GeneratorOutput,
    ReflectorOutput,
    CuratorOutput,
    # Modular reflector
    ModularReflector,
    ModularReflectorOutput,
    MutableUpdate,
    build_modular_excerpt,
    MODULAR_REFLECTOR_PROMPT,
)
from .adaptation import (
    OfflineAdapter,
    OnlineAdapter,
    Sample,
    TaskEnvironment,
    EnvironmentResult,
    AdapterStepResult,
)
from .retrieval import (
    SemanticRetriever,
    RetrievalResult,
    # Modular retriever
    ModularSemanticRetriever,
    ModularRetrievalResult,
)

__all__ = [
    # Traditional Playbook
    "Bullet",
    "Playbook",
    # Modular Playbook
    "ModularBullet",
    "ModularPlaybook",
    "FixedModules",
    "MutableModules",
    "ContextualStates",
    "DecisionBehaviors",
    "Uncertainty",
    "DelayedAssumptions",
    # Delta
    "DeltaOperation",
    "DeltaBatch",
    # LLM
    "LLMClient",
    "DummyLLMClient",
    "UniversalLLMClient",
    # Traditional roles
    "Reflector",
    "Curator",
    "GeneratorOutput",
    "ReflectorOutput",
    "CuratorOutput",
    # Modular roles
    "ModularReflector",
    "ModularReflectorOutput",
    "MutableUpdate",
    "build_modular_excerpt",
    "MODULAR_REFLECTOR_PROMPT",
    # Adapters
    "OfflineAdapter",
    "OnlineAdapter",
    "Sample",
    "TaskEnvironment",
    "EnvironmentResult",
    "AdapterStepResult",
    # Retrievers
    "SemanticRetriever",
    "RetrievalResult",
    "ModularSemanticRetriever",
    "ModularRetrievalResult",
]
