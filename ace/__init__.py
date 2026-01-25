"""Agentic Context Engineering (ACE) reproduction framework."""

from .playbook import (
    Bullet,
    Playbook,
    # 模块化经验库
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
    Generator,
    Reflector,
    Curator,
    GeneratorOutput,
    ReflectorOutput,
    CuratorOutput,
    # 模块化反思器
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
    # 模块化检索器
    ModularSemanticRetriever,
    ModularRetrievalResult,
)

__all__ = [
    # 传统 Playbook
    "Bullet",
    "Playbook",
    # 模块化 Playbook
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
    # 传统角色
    "Generator",
    "Reflector",
    "Curator",
    "GeneratorOutput",
    "ReflectorOutput",
    "CuratorOutput",
    # 模块化角色
    "ModularReflector",
    "ModularReflectorOutput",
    "MutableUpdate",
    "build_modular_excerpt",
    "MODULAR_REFLECTOR_PROMPT",
    # 适应器
    "OfflineAdapter",
    "OnlineAdapter",
    "Sample",
    "TaskEnvironment",
    "EnvironmentResult",
    "AdapterStepResult",
    # 检索器
    "SemanticRetriever",
    "RetrievalResult",
    "ModularSemanticRetriever",
    "ModularRetrievalResult",
]
