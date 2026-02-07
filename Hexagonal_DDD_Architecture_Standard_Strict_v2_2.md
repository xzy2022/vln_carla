# Hexagonal + DDD Architecture Standard (Strict, v2.2)

## Purpose
Define **strict, production-grade rules** for building systems using
**Hexagonal Architecture + DDD + Clean layering** under **Pylance `strict`** mode.

This is an **implementation standard**, not a conceptual discussion.
All dependency, ownership, placement, and typing rules are **hard constraints**.

---

## 1. Layered Structure (Default)

```text
- domain/            # Core business model: Entities, Value Objects, Aggregates, Domain Services
- usecases/          # Application layer: Use Cases (Application Services), orchestration, app-level ports
- adapters/          # Primary (driving) adapters: HTTP, GraphQL, CLI, MQ consumers, presenters
- infrastructure/    # Secondary (driven) technical adapters: DB, cache, MQ producer, external clients
- app/               # Composition Root only: DI wiring, config, bootstrap
```

Notes:
- In Hexagonal terminology, both `adapters/` and `infrastructure/` are adapters.
- This split is intentional:
  - `adapters/` -> inbound protocols and presentation
  - `infrastructure/` -> outbound technical integrations

---

## 2. Dependency Rules (Hard Constraints)

### domain/
- MUST NOT import: `usecases/`, `adapters/`, `infrastructure/`, `app/`
- Contains:
  - Entities, Value Objects, Aggregates
  - Domain Services (pure business logic only)
  - Domain Events
- MUST:
  - Contain no framework, transport, persistence, or IO code
  - Remain persistence-ignorant
  - Express business invariants only

### usecases/
- MAY import: `domain/`
- MUST NOT import: `adapters/`, `infrastructure/`, `app/`
- Contains:
  - Use Case input ports (interfaces)
  - Application Services / Interactors
  - Input / Output DTOs
  - Application-owned ports
  - Transaction boundary coordination
- MUST:
  - Depend only on abstractions
  - Contain no transport or persistence implementations

### adapters/
- MAY import: `usecases/`
- MUST NOT import: `infrastructure/`, `app/`
- Contains:
  - Controllers / Resolvers / CLI handlers / MQ consumers
  - Presenters / Response mappers
- MUST:
  - Translate protocol input -> use case input DTO
  - Invoke use case input ports only
  - Use DTOs, not domain objects, at boundaries

### infrastructure/
- MAY import:
  - `domain/`
  - `domain/ports`
  - `usecases/ports`
- MUST NOT import:
  - `adapters/`, `app/`
  - Concrete use case implementations
- Contains:
  - Repository implementations
  - External API clients
  - Message publishers
  - Technical frameworks and drivers

### app/
- MAY import all layers
- Responsible ONLY for:
  - Dependency wiring
  - Configuration
  - Process startup
  - Lifecycle management
- MUST NOT contain business or application logic

---

## 3. Ports Ownership Rules

### domain/ports
Used for domain-owned contracts:
- Aggregate repositories
- Domain event publishers
- Domain policies

### usecases/ports
Used for application-owned dependencies:
- Notifications
- External workflow coordination
- Integration hooks
- Unit of Work / Transaction management
- Read-model gateways

Rules:
- Domain and usecases depend only on abstractions
- Infrastructure contains implementations only
- If unclear, choose the innermost valid layer

---

## 4. Use Case Contracts

- Every use case MUST expose an input port (interface)
- Input ports are defined in `usecases/`
- Adapters depend ONLY on these interfaces

### DTO Rules
- Input and output DTOs are defined in `usecases/`
- Adapters MUST NOT expose domain entities or value objects

---

## 5. Adapter Responsibilities

### Primary Adapters (`adapters/`)
- Handle inbound protocols (HTTP, GraphQL, CLI, MQ)
- Validate, authorize, map DTOs, invoke use cases

### Output Mapping
- Presenters / response mappers
- Convert output DTOs -> protocol responses
- MUST NOT invoke use cases

### Secondary Adapters (`infrastructure/`)
- Persistence, external clients, publishers
- Technical policies (retry, timeout, serialization)

---

## 6. Transaction & Unit of Work

- Transaction boundaries are application concerns
- Interfaces live in `usecases/ports` or `domain/ports`
- Implementations live in `infrastructure/`
- Use cases coordinate transactions explicitly

---

## 7. Domain Service vs Application Service

- Domain Services:
  - Pure business logic only
  - No IO, no orchestration, no external calls
- Application Services:
  - Orchestrate aggregates
  - Call outbound ports
  - Manage transactions

---

## 8. CQRS Compatibility

### Command Path
- State changes MUST go through aggregates
- Enforce invariants and emit domain events

### Query Path
- Read models may bypass aggregates
- Must be side-effect free
- Must not live in `domain/`

---

## 9. Explicit Non-Goals

- Do NOT collapse layers
- Do NOT put technical code in `domain/` or `usecases/`
- Do NOT leak transport schemas into domain
- Do NOT break dependency inversion

---

## 10. Scaling & Packaging

For larger systems:

```text
contexts/<context_name>/
  - domain/
  - usecases/
  - adapters/
  - infrastructure/
```

---

## 11. Code Generation Expectations

- Show exact folder placement
- Respect all dependency constraints
- Keep contracts explicit and small
- Prefer clarity over brevity
- When unsure, preserve domain purity

---

## 12. Pylance Strict Typing Rules (Mandatory)

This section defines **mandatory typing constraints** that are enforced together with
the architectural boundaries defined above.

Typing rules are **not stylistic**. They are part of the architecture.
Violating typing boundaries is considered equivalent to violating layer boundaries.

---

### 12.0 Tooling Baseline

- `python.analysis.typeCheckingMode = "strict"` MUST be enabled.
- A single `pyrightconfig.json` (or workspace-level Pylance config) MUST be the source of truth.
- CI and local development MUST use the same strict configuration.
- `pythonVersion` MUST be explicitly declared to avoid divergent diagnostics.

---

### 12.1 Annotation Coverage

- All public **and internal** functions and methods MUST declare:
  - Parameter types
  - Return types
- Class attributes MUST be explicitly typed.
- Module-level constants and variables MUST be explicitly typed when inference is ambiguous.
- Empty collections MUST declare concrete generic types.

Clarifications:
- “Internal” means any function/method that is not strictly private to a single function scope.
- `@dataclass` fields MUST be typed; generated `__init__` signatures MAY rely on those annotations.
- `@property` methods MUST declare return types.
- Lambdas MUST NOT be used for logic that would require non-trivial typing.

---

### 12.2 `Any` Policy

- `Any` is **strictly prohibited** in:
  - `domain/`
  - `usecases/`
- In `adapters/` and `infrastructure/`, `Any` is allowed **only at external boundaries** and MUST be narrowed immediately.
- `Any` MUST NEVER:
  - Appear in DTOs
  - Appear in ports or interfaces
  - Propagate across layer boundaries

Allowed narrowing mechanisms (explicit whitelist):
- `TypedDict`, `dataclass`, `attrs`, or equivalent typed models
- `Protocol` for behavioral contracts
- `TypeGuard`
- `isinstance` / explicit runtime validation

Prohibited:
- Using `cast()` as a substitute for validation
- Passing raw `dict[str, Any]` or untyped payloads into `usecases/` or `domain/`

---

### 12.3 Type Compatibility and Narrowing

- Variable reassignment MUST remain type-compatible.
- `Union` types MUST be narrowed before use.
- `Optional[T]` MUST be explicitly checked before dereference.
- All code paths MUST return values matching the declared return type.

Architectural rule:
- All dynamic or partially-typed input MUST be fully validated and narrowed **inside adapters**.
- Use cases MUST receive fully-typed DTOs and MUST NOT perform input sanitation.

---

### 12.4 Access and Encapsulation

- Cross-module access to private or protected members is forbidden.
- Layer boundaries MUST be crossed using:
  - `Protocol`s or
  - Abstract Base Classes (ABCs)

Concrete implementations MUST NOT be referenced across layers.
This applies to **types as well as runtime values**.

---

### 12.5 Generic and Collection Safety

- All generic containers (`list`, `dict`, `set`, `tuple`, `Mapping`, etc.) MUST specify concrete type parameters.
- DTOs and ports MUST NOT use:
  - `dict[str, object]`
  - `Mapping[str, object]`
  - Untyped or weakly-typed payload containers

Guideline:
- Prefer immutable or read-only collection types in contracts when mutation is not required.

---

### 12.6 Exhaustiveness and Completeness

- Branching on `Enum`, `Literal`, or discriminated unions SHOULD be exhaustive.
- Use `typing.assert_never` (or `typing_extensions.assert_never`) for unreachable branches.
- Error-handling paths MUST preserve declared return contracts.

---

### 12.7 Layer-Specific Typing Requirements

#### domain/
- No `Any`.
- No runtime-typed objects.
- `cast()` is forbidden **except** when:
  - A runtime invariant is explicitly validated, AND
  - A comment documents the invariant, its owner, and why the cast is safe.

#### usecases/
- All ports and DTOs MUST be fully typed.
- No implicit `dict`-based contracts.
- Use cases depend only on abstract, typed interfaces.

#### adapters/
- Boundary parsing MAY be dynamic.
- All data handed off to `usecases/` MUST be strict DTO types.
- No framework/request objects may cross the boundary.

#### infrastructure/
- Third-party or untyped responses MUST be converted to typed models immediately.
- These technical models MUST remain inside `infrastructure/`.
- Only domain or usecase-defined abstractions may cross outward.

#### app/
- DI factories/providers MUST declare precise interface return types.
- No `Any`-typed service locators or containers.

---

### 12.8 Definition of Done (Type Quality Gate)

A change is considered complete only if:

- Pylance strict diagnostics report **zero errors** in changed modules.
- No new `# type: ignore` is introduced without:
  - A rule-specific ignore (e.g. `# type: ignore[arg-type]`)
  - A rationale comment explaining why it is safe

Recommended (when applicable):
- Link to a tracking issue or TODO with owner and follow-up plan.

---

### 12.9 Architectural Typing Boundary Rule

- `domain/` and `usecases/` MUST NOT reference:
  - Types
  - Schemas
  - Models
defined in `adapters/` or `infrastructure/`, even for typing purposes.

All cross-layer typing MUST flow through:
- Domain ports
- Use case ports
- Use case DTOs

Typing boundaries and architectural boundaries are enforced together.
