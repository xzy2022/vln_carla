Hexagonal + DDD Architecture Standard (v2)

Purpose:
Define strict, production-oriented rules for building systems with
Hexagonal Architecture + DDD + Clean layering.

This is an implementation standard, not a conceptual discussion.
Follow the dependency and ownership rules exactly.

========================================
1. Layered Structure (Default)
========================================

The project default structure is:

- domain/            # Core business model: Entities, Value Objects, Aggregates, Domain Services
- usecases/          # Application layer: Use Cases (Application Services), orchestration, app-level ports
- adapters/          # Primary (driving) adapters: HTTP, GraphQL, CLI, MQ consumers, presenters
- infrastructure/    # Secondary (driven) technical adapters: DB, cache, MQ producer, external clients
- app/               # Composition Root only: DI wiring, config, bootstrap

Note:
- In Hexagonal terminology, both adapters/ and infrastructure/ are "adapters".
- This split is intentional for engineering clarity:
  - adapters/ focuses on inbound protocol handling.
  - infrastructure/ focuses on outbound technical integrations.

========================================
2. Dependency Rules (Hard Constraints)
========================================

- domain/
  - MUST NOT import usecases/, adapters/, infrastructure/, app/
  - Contains NO framework, transport, or IO code
  - Must stay persistence-ignorant

- usecases/
  - MAY import domain/
  - MUST NOT import adapters/, infrastructure/, app/
  - Coordinates use case flow and transaction boundaries
  - Must not contain transport or persistence implementation details

- adapters/
  - MAY import usecases/
  - SHOULD avoid direct domain exposure at API boundaries (prefer DTOs)
  - MUST NOT import infrastructure/ or app/
  - Contains protocol-specific mapping, validation, and invocation

- infrastructure/
  - MAY import domain/ and usecases/ port interfaces
  - MUST NOT import adapters/ or app/
  - Implements repositories, gateways, publishers, external service clients
  - Contains technical frameworks and drivers

- app/
  - MAY import all layers
  - Responsible ONLY for wiring, configuration, process startup, and lifecycle
  - Contains NO business rules

========================================
3. Ports Ownership Rules
========================================

Port placement follows business ownership:

- domain/ports
  - For domain-owned contracts (for example repository abstractions tied to aggregates)
  - Use when the contract is part of core domain semantics

- usecases/ports
  - For application-owned outbound dependencies
  - For cross-context orchestration concerns (notification, remote workflow, integration hooks)

Rules:
- infrastructure/ contains only implementations of domain/ports and usecases/ports.
- domain/ and usecases/ must depend on abstractions, never concrete adapters.
- If ownership is unclear, choose the innermost valid layer.

========================================
4. Adapter Responsibilities
========================================

Primary (driving) adapters in adapters/:
- Controllers / Resolvers / CLI handlers / MQ consumers
- Translate transport protocol into use case input DTOs
- Invoke use cases

Output mapping in adapters/:
- Presenters / Response mappers
- Format use case outputs for transport-specific responses
- Must NOT invoke use cases

Secondary (driven) adapters in infrastructure/:
- Persistence implementations
- External API clients
- Message publishers
- Technical policy implementations (retry, timeout, serialization)

========================================
5. CQRS Compatibility Rule
========================================

For command paths:
- Writes that change business state MUST go through domain aggregates and domain rules.

For query paths:
- Read models MAY bypass aggregate rehydration for performance.
- Such read paths MUST remain side-effect free and must not violate domain invariants.

========================================
6. Explicit Non-Goals
========================================

- Do NOT collapse layers for convenience.
- Do NOT place outbound technical implementations in usecases/ or domain/.
- Do NOT access DB, network, filesystem, or frameworks from usecases/ or domain/.
- Do NOT create framework-driven coupling that breaks dependency inversion.
- Do NOT leak transport schemas directly into domain model.

========================================
7. Scaling and Packaging Guidance
========================================

For larger systems, keep the same layering per bounded context/feature, for example:
- contexts/<context_name>/domain
- contexts/<context_name>/usecases
- contexts/<context_name>/adapters
- contexts/<context_name>/infrastructure

This prevents a single global layer folder from becoming a coupling hotspot.

========================================
8. Output Expectations for Code Generation
========================================

When generating code or structure:
- Show exact folder placement for each artifact.
- Respect all dependency constraints.
- Keep contracts explicit and small.
- Prefer clarity over brevity.
- If uncertain, choose the option that preserves dependency inversion and domain purity.
