# HTTQ / HTTQS: Hyper-Text Transport Quantum

The **Lucia Browser** natively supports the **HTTQ** and **HTTQS** protocols, the foundational transport layers of the Nara Web 4.0 ecosystem. These protocols replace traditional TCP/IP addressing with cryptographic identity routing.

## 1. What is HTTQ?
**HTTQ (Hyper-Text Transport Quantum)** is a decentralized, peer-to-peer transport protocol that routes requests based on **Quantum Identity Keys (QIK)** rather than IP addresses.

### Key Features:
- **Identity-Based Routing**: Data is routed securely to a specific cryptographic identity, regardless of its physical location or IP address.
- **Mesh Propagation**: Requests traverse the **AevIP Mesh**, utilizing optimal paths calculated by the **Nara Engine**.
- **Censorship Resistance**: By decoupling content from location, HTTQ ensures data availability even if individual nodes are blocked.

---

## 2. HTTQS (Secure Transport)
**HTTQS** adds an additional layer of end-to-end encryption using **Quantum-Resistant Cryptography (QRC)**.

### Security Model:
- **Handshake**: A Diffie-Hellman key exchange secured by post-quantum algorithms (e.g., Kyber/Dilithium).
- **Session Keys**: Ephemeral keys are generated for each session, ensuring perfect forward secrecy.
- **Privacy**: Traffic analysis is mitigated by **Tansu Relays**, which obfuscate the origin and destination of packets.

---

## 3. URL Structure
Nara Web 4.0 URLs follow this format:

```
httq://[orbital-identity]/[resource-path]
```

- **Scheme**: `httq://` or `httqs://`
- **Orbital Identity**: A `.q` domain (e.g., `wakanda.q`) or a raw public key hash.
- **Resource Path**: Standard path to the requested content or service.

### Example:
`httqs://wakanda.q/manifesto.md`

1.  **Resolution**: The browser queries the **QNS (Quantum Name System)** to resolve `wakanda.q` to its current orbital identity hash.
2.  **Routing**: The request is routed through the mesh to the nearest node hosting that identity.
3.  **Transport**: Data is retrieved securely via HTTQS.

---

## 4. Integration
Lucia handles these protocols natively via the **Nara Engine**. Developers can build dApps that utilize HTTQ without needing to manage the underlying cryptographic complexity.

**See Also**: [ARCHITECTURE.md](ARCHITECTURE.md) for engine details.

---
© 2026 WPWakanda LLC / Aevov AI Technologies.
*Secure the mesh.* 🔒QM
