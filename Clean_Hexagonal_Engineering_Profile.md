You are designing code using a strict, engineering-oriented variant of
Clean Architecture + Hexagonal Architecture + DDD application layering.

This is NOT a conceptual discussion.
You must follow the structural and dependency rules below exactly.
Do not invent alternative interpretations.

========================================
1. Layered Structure (Fixed)
========================================

The project MUST be structured as:

- domain/            # Enterprise business rules (Entities, Value Objects, Domain Services)
- usecases/          # Application business rules (Use Cases + Ports)
- adapters/          # Interface adapters (Controllers, Presenters)
- infrastructure/    # Technical implementations of ports (DB, MQ, external systems)
- app/               # Composition Root (dependency injection, bootstrap only)

========================================
2. Dependency Rules (Hard Constraints)
========================================

- domain/
  - MUST NOT import usecases/, adapters/, infrastructure/, app/
  - Contains NO framework or IO code

- usecases/
  - MAY import domain/
  - MUST NOT import adapters/, infrastructure/, app/
  - Defines ports (interfaces) for all external dependencies

- adapters/
  - MAY import usecases/ and domain/
  - MUST NOT import infrastructure/ or app/
  - Contains protocol-specific adapters only (HTTP, GraphQL, CLI, MQ)

- infrastructure/
  - MAY import usecases/ (ports/interfaces) and domain/
  - MUST NOT import adapters/ or app/
  - Implements ports defined in usecases/ or domain/

- app/
  - MAY import ALL layers
  - Responsible ONLY for dependency wiring, configuration, and application bootstrap
  - Contains NO business logic

========================================
3. Ports & Gateways Rules
========================================

- All ports (Repository, Gateway, Publisher, External Service interfaces)
  MUST be defined in usecases/ports (or domain/ports if required by domain logic).

- infrastructure/ contains ONLY implementations of those ports.

- No use case or domain code may depend on concrete implementations.

========================================
4. Adapters Responsibilities
========================================

- Adapters are split by protocol:
  - adapters/http/
  - adapters/graphql/
  - adapters/cli/
  - adapters/mq/

- Inside each adapter:
  - Controllers / Resolvers / Consumers are INPUT adapters (driving side)
  - Presenters / Response Mappers are OUTPUT adapters (driven side)

- Controllers trigger use cases.
- Presenters format use case output.
- Presenters do NOT trigger use cases.

========================================
5. Explicit Non-Goals
========================================

- Do NOT collapse layers.
- Do NOT place infrastructure code in adapters.
- Do NOT access databases, frameworks, or external APIs from usecases or domain.
- Do NOT invent "shortcuts" or framework-driven coupling.

========================================
6. Output Expectations
========================================

When generating code or structure:
- Show clear folder placement.
- Respect all dependency rules.
- Prefer clarity over brevity.
- If unsure, choose the option that preserves dependency inversion.
