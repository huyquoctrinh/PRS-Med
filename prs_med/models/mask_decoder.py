from segment_model.mask_decoder_v5 import PromptedMaskDecoder as LegacyPromptedMaskDecoder


class PromptedMaskDecoder(LegacyPromptedMaskDecoder):
    """Thin alias for the legacy PromptedMaskDecoder.

    This wrapper allows future extensions while keeping backward compatibility.
    """

    pass


