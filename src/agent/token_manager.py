"""
Gerenciador de tokens para GitHub Models

Centraliza a lógica de rotação de tokens para evitar duplicação
entre main.py e llm_eval.py.

FLUXO:
Token ativo → Chamada → Contabiliza → Esgotou cota? → Próximo token
"""

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential


class GerenciadorTokens:
    """
    Gerencia múltiplos tokens GitHub com rotação automática ao atingir cota diária.

    Uso:
        gm = GerenciadorTokens(tokens=[tok1, tok2], endpoint=..., limite_diario=50)
        cliente = gm.cliente_atual()   # lança RuntimeError se todos esgotados
        gm.registrar_chamada()
        gm.marcar_esgotado()           # em caso de rate limit 429
    """

    def __init__(
        self,
        tokens: list[str],
        endpoint: str,
        limite_diario: int = 50,
        timeout: float = 60.0,
    ):
        """
        Args:
            tokens: lista de tokens (strings vazias são ignoradas)
            endpoint: URL da API (ex: config.GITHUB_ENDPOINT)
            limite_diario: chamadas permitidas por token por dia
            timeout: timeout em segundos para cada chamada HTTP
        """
        self._limite_diario = limite_diario
        self._timeout = timeout
        self._clients: list[ChatCompletionsClient] = []
        self._calls:   list[int] = []

        for token in tokens:
            if token:
                self._clients.append(
                    ChatCompletionsClient(
                        endpoint=endpoint,
                        credential=AzureKeyCredential(token),
                        retry_total=0,
                        connection_timeout=timeout,
                        read_timeout=timeout,
                    )
                )
                self._calls.append(0)

        if not self._clients:
            raise RuntimeError("Nenhum GITHUB_TOKEN válido configurado")

        self._idx = 0  # índice do token ativo

    # ─────────────────────────────────────────────────────────────
    # API pública
    # ─────────────────────────────────────────────────────────────

    def cliente_atual(self) -> ChatCompletionsClient:
        """
        Retorna o cliente do token ativo.
        Avança para o próximo token se o atual estiver esgotado.
        Lança RuntimeError se todos os tokens estão esgotados.
        """
        for _ in range(len(self._clients)):
            if self._calls[self._idx] < self._limite_diario:
                return self._clients[self._idx]
            print(f"  ⚠️  Token {self._idx + 1} esgotado ({self._limite_diario} req/dia) — alternando")
            self._idx = (self._idx + 1) % len(self._clients)

        raise RuntimeError("Todos os tokens GitHub esgotaram a cota diária")

    def registrar_chamada(self) -> int:
        """Incrementa o contador do token ativo. Retorna o total acumulado."""
        self._calls[self._idx] += 1
        return self._calls[self._idx]

    def marcar_esgotado(self) -> bool:
        """
        Marca o token atual como esgotado (após erro 429) e avança para o próximo.
        Retorna True se ainda há tokens disponíveis, False se todos esgotaram.
        """
        print(f"  ⚠️  Rate limit — marcando token {self._idx + 1} como esgotado")
        self._calls[self._idx] = self._limite_diario
        self._idx = (self._idx + 1) % len(self._clients)
        return not all(c >= self._limite_diario for c in self._calls)

    def todos_esgotados(self) -> bool:
        """Retorna True se todos os tokens atingiram a cota diária."""
        return all(c >= self._limite_diario for c in self._calls)

    @property
    def idx(self) -> int:
        """Índice do token ativo (base 0)."""
        return self._idx

    @property
    def limite_diario(self) -> int:
        return self._limite_diario

    @property
    def timeout(self) -> float:
        return self._timeout

    def status(self) -> str:
        """Resumo legível do estado de cada token."""
        partes = [
            f"Token {i+1}: {c}/{self._limite_diario}"
            for i, c in enumerate(self._calls)
        ]
        return "  |  ".join(partes)
