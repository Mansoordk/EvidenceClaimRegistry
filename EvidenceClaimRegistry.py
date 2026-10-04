# { "Depends": "py-genlayer:1jb45aa8ynh2a9c9xn3b7qqh8sm5q93hwfp7jqmwsfhh8jpz09h6" }

from genlayer import *
import json


MAX_SOURCES = 3
MAX_SOURCE_CHARS = 7000

VERIFIED = "VERIFIED"
REFUTED = "REFUTED"
INCONCLUSIVE = "INCONCLUSIVE"


class EvidenceClaimRegistry(gl.Contract):
    """
    EvidenceClaimRegistry

    A reusable GenLayer consensus primitive for verifying factual
    claims against public web evidence.

    Users register a claim with up to three public evidence URLs.
    GenLayer validators independently retrieve and evaluate the
    evidence, then reach consensus on the substantive verdict.

    Possible verdicts:
        VERIFIED
        REFUTED
        INCONCLUSIVE
    """

    # ---------------------------------------------------------
    # Persistent state
    # ---------------------------------------------------------

    claim_texts: TreeMap[str, str]

    source_1: TreeMap[str, str]
    source_2: TreeMap[str, str]
    source_3: TreeMap[str, str]

    verdicts: TreeMap[str, str]
    confidences: TreeMap[str, u8]
    evidence_summaries: TreeMap[str, str]

    resolved: TreeMap[str, bool]

    claim_ids: DynArray[str]

    # ---------------------------------------------------------
    # Constructor
    # ---------------------------------------------------------

    def __init__(self):
        pass

    # ---------------------------------------------------------
    # Register a new claim
    # ---------------------------------------------------------

    @gl.public.write
    def register_claim(
        self,
        claim_id: str,
        claim_text: str,
        source1: str,
        source2: str,
        source3: str,
    ):
        if not claim_id:
            raise gl.vm.UserError(
                "claim_id cannot be empty"
            )

        if len(claim_id) > 96:
            raise gl.vm.UserError(
                "claim_id is too long"
            )

        if not claim_text:
            raise gl.vm.UserError(
                "claim_text cannot be empty"
            )

        if len(claim_text) > 1000:
            raise gl.vm.UserError(
                "claim_text is too long"
            )

        if claim_id in self.claim_texts:
            raise gl.vm.UserError(
                "claim_id already exists"
            )

        # At least one source is required.
        if not source1:
            raise gl.vm.UserError(
                "source1 is required"
            )

        sources = [source1]

        if source2:
            sources.append(source2)

        if source3:
            sources.append(source3)

        if len(sources) > MAX_SOURCES:
            raise gl.vm.UserError(
                "maximum of three sources allowed"
            )

        # Validate URLs.
        for url in sources:

            if (
                not url.startswith("http://")
                and not url.startswith("https://")
            ):
                raise gl.vm.UserError(
                    "all sources must use http:// or https://"
                )

            if len(url) > 500:
                raise gl.vm.UserError(
                    "source URL is too long"
                )

        # Store claim.
        self.claim_texts[claim_id] = claim_text

        self.source_1[claim_id] = source1
        self.source_2[claim_id] = source2
        self.source_3[claim_id] = source3

        self.verdicts[claim_id] = INCONCLUSIVE
        self.confidences[claim_id] = u8(0)
        self.evidence_summaries[claim_id] = ""

        self.resolved[claim_id] = False

        self.claim_ids.append(claim_id)

    # ---------------------------------------------------------
    # Verify claim using GenLayer consensus
    # ---------------------------------------------------------

    @gl.public.write
    def verify_claim(
        self,
        claim_id: str,
    ):

        if claim_id not in self.claim_texts:
            raise gl.vm.UserError(
                "claim does not exist"
            )

        if self.resolved[claim_id]:
            raise gl.vm.UserError(
                "claim has already been resolved"
            )

        claim_text = self.claim_texts[claim_id]

        source1 = self.source_1[claim_id]
        source2 = self.source_2[claim_id]
        source3 = self.source_3[claim_id]

        sources = [source1]

        if source2:
            sources.append(source2)

        if source3:
            sources.append(source3)

        # -----------------------------------------------------
        # Leader execution
        # -----------------------------------------------------

        def leader_fn():

            evidence_parts = []

            successful_sources = 0

            for index, url in enumerate(sources):

                try:

                    response = gl.nondet.web.get(url)

                    body = response.body.decode("utf-8")

                    if body:

                        successful_sources += 1

                        evidence_parts.append(
                            "SOURCE "
                            + str(index + 1)
                            + " URL: "
                            + url
                            + "\n"
                            + body[:MAX_SOURCE_CHARS]
                        )

                    else:

                        evidence_parts.append(
                            "SOURCE "
                            + str(index + 1)
                            + " URL: "
                            + url
                            + "\nEMPTY RESPONSE"
                        )

                except Exception:

                    evidence_parts.append(
                        "SOURCE "
                        + str(index + 1)
                        + " URL: "
                        + url
                        + "\nSOURCE UNAVAILABLE"
                    )

            evidence = "\n\n".join(evidence_parts)

            prompt = f"""
You are an evidence verification agent operating inside
a GenLayer Intelligent Contract.

Your job is to determine whether the CLAIM is supported
or contradicted by the supplied WEB EVIDENCE.

CLAIM:
{claim_text}

WEB EVIDENCE:
{evidence}

RULES:

1. Use only the supplied evidence.
2. Do not use outside knowledge.
3. VERIFIED means the evidence materially supports the claim.
4. REFUTED means the evidence materially contradicts the claim.
5. INCONCLUSIVE means the evidence is insufficient,
   ambiguous, unavailable, or contradictory.
6. Do not infer facts that are not supported by the evidence.
7. confidence must be an integer from 0 to 100.
8. source_count must represent the number of sources
   containing useful evidence.
9. Keep evidence_summary concise and factual.

Return JSON with exactly these fields:

{{
    "verdict": "VERIFIED",
    "confidence": 0,
    "source_count": 0,
    "evidence_summary": "short evidence-based explanation"
}}

The verdict MUST be exactly one of:

VERIFIED
REFUTED
INCONCLUSIVE
"""

            result = gl.nondet.exec_prompt(
                prompt,
                response_format="json",
            )

            # -------------------------------------------------
            # Validate leader result
            # -------------------------------------------------

            if not isinstance(result, dict):
                raise gl.vm.UserError(
                    "LLM result is not an object"
                )

            if "verdict" not in result:
                raise gl.vm.UserError(
                    "LLM result missing verdict"
                )

            if "confidence" not in result:
                raise gl.vm.UserError(
                    "LLM result missing confidence"
                )

            if "source_count" not in result:
                raise gl.vm.UserError(
                    "LLM result missing source_count"
                )

            if "evidence_summary" not in result:
                raise gl.vm.UserError(
                    "LLM result missing evidence_summary"
                )

            verdict = result["verdict"]

            if verdict not in [
                VERIFIED,
                REFUTED,
                INCONCLUSIVE,
            ]:
                raise gl.vm.UserError(
                    "invalid verdict"
                )

            confidence = result["confidence"]

            if not isinstance(confidence, int):
                raise gl.vm.UserError(
                    "confidence must be an integer"
                )

            if confidence < 0 or confidence > 100:
                raise gl.vm.UserError(
                    "confidence must be between 0 and 100"
                )

            source_count = result["source_count"]

            if not isinstance(source_count, int):
                raise gl.vm.UserError(
                    "source_count must be an integer"
                )

            if source_count < 0 or source_count > len(sources):
                raise gl.vm.UserError(
                    "invalid source_count"
                )

            summary = result["evidence_summary"]

            if not isinstance(summary, str):
                raise gl.vm.UserError(
                    "evidence_summary must be a string"
                )

            return {
                "verdict": verdict,
                "confidence": confidence,
                "source_count": source_count,
                "evidence_summary": summary[:1200],
                "successful_sources": successful_sources,
            }

        # -----------------------------------------------------
        # Validator execution
        # -----------------------------------------------------

        def validator_fn(leader_result):

            try:

                if not isinstance(
                    leader_result,
                    gl.vm.Return
                ):
                    return False

                leader = leader_result.calldata

                # Validator independently repeats the entire
                # web retrieval + reasoning process.
                validator = leader_fn()

                # -------------------------------------------------
                # Substantive equivalence rules
                # -------------------------------------------------

                # The core factual verdict must agree.
                if (
                    leader["verdict"]
                    != validator["verdict"]
                ):
                    return False

                # Validators must agree on how many sources
                # actually provided useful evidence.
                if (
                    leader["source_count"]
                    != validator["source_count"]
                ):
                    return False

                # Confidence can naturally vary between LLM
                # executions, so allow a controlled tolerance.
                confidence_difference = abs(
                    leader["confidence"]
                    - validator["confidence"]
                )

                if confidence_difference > 15:
                    return False

                # Both executions must have retrieved evidence.
                if (
                    leader["successful_sources"] == 0
                    and validator["successful_sources"] == 0
                ):
                    return False

                return True

            except Exception:

                # Any validator failure means disagreement.
                return False

        # -----------------------------------------------------
        # Execute consensus
        # -----------------------------------------------------

        result = gl.vm.run_nondet_unsafe(
            leader_fn,
            validator_fn,
        )

        # -----------------------------------------------------
        # Persist ONLY the consensus-approved result.
        # -----------------------------------------------------

        self.verdicts[claim_id] = result["verdict"]

        self.confidences[claim_id] = u8(
            result["confidence"]
        )

        self.evidence_summaries[claim_id] = (
            result["evidence_summary"]
        )

        self.resolved[claim_id] = True

    # ---------------------------------------------------------
    # Read claim
    # ---------------------------------------------------------

    @gl.public.view
    def get_claim(
        self,
        claim_id: str,
    ):

        if claim_id not in self.claim_texts:
            raise gl.vm.UserError(
                "claim does not exist"
            )

        return {
            "claim_id": claim_id,

            "claim_text":
                self.claim_texts[claim_id],

            "source1":
                self.source_1[claim_id],

            "source2":
                self.source_2[claim_id],

            "source3":
                self.source_3[claim_id],

            "verdict":
                self.verdicts[claim_id],

            "confidence":
                self.confidences[claim_id],

            "evidence_summary":
                self.evidence_summaries[claim_id],

            "resolved":
                self.resolved[claim_id],
        }

    # ---------------------------------------------------------
    # Get verdict
    # ---------------------------------------------------------

    @gl.public.view
    def get_verdict(
        self,
        claim_id: str,
    ) -> str:

        if claim_id not in self.claim_texts:
            raise gl.vm.UserError(
                "claim does not exist"
            )

        return self.verdicts[claim_id]

    # ---------------------------------------------------------
    # Check resolution
    # ---------------------------------------------------------

    @gl.public.view
    def is_resolved(
        self,
        claim_id: str,
    ) -> bool:

        if claim_id not in self.claim_texts:
            return False

        return self.resolved[claim_id]

    # ---------------------------------------------------------
    # Number of claims
    # ---------------------------------------------------------

    @gl.public.view
    def get_claim_count(self) -> u256:
        return u256(len(self.claim_ids))

    # ---------------------------------------------------------
    # Claim IDs
    # ---------------------------------------------------------

    @gl.public.view
    def get_claim_ids(self) -> DynArray[str]:
        return self.claim_ids
