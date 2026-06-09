/// Substrings (lowercased) — any retrieved chunk containing one is dropped
/// before it reaches the LLM, preventing scene-specific content from surfacing
/// in character descriptions. Add a new entry to DEFAULT_SCENE_MARKERS and
/// rebuild to make it permanent; set SCENE_FILTER_MARKERS in the environment
/// to override for a specific deployment without a code change.
const DEFAULT_SCENE_MARKERS: &[&str] = &[
    "only the dwarves in doragon know",
    "worshiped by snake people",
    "forgive me, i need a moment alone",
    "screech of a wild animal",
    "grapples with an internal struggle as he looks at you",
    "may i kiss",
    "returning your kiss",
    "returns your kiss",
    "fade to black",
    "brushing a strand of your hair",
    "powerful lover",
    "pour his soul into",
    "eternity is meaningless without you",
    "kneels to pray",
    "long gashes marking his back",
    "whip stained with blood",
    "whip wet with blood",
    "iron gauntlet",
    "scirocco he won't look",
    "branchwag",
    "today at",
    "if florian is brought up",
    "~100 years? or",
    "are — 11/",
    "are — 07/",
    "he's referring to florian",
    "what is the red dragon's name",
    "he would refer to the red dragon as his brother",
    "if they decide to go for the vampire dungeon",
    "lead them to the dwarves of doragon",
    "bat-like wings extending from her back",
];

/// Rules appended verbatim as bullet points to the LLM system prompt.
/// Each entry is a complete, self-contained instruction. Add a new entry to
/// DEFAULT_PROMPT_RULES and rebuild to make it permanent; set PROMPT_EXTRA_RULES
/// (pipe-separated) in the environment to override for a specific deployment.
const DEFAULT_PROMPT_RULES: &[&str] = &[
    "Never write character names in all capitals. Always use normal title case — \
     write \"Alora\", not \"ALORA\"; \"Florian\", not \"FLORIAN\".",

    "Do not describe a character's private acts of guilt, self-punishment, or personal \
     religious ritual. If guilt or regret is relevant, state that the character carries \
     a burden or feels remorse; do not describe the specific form that takes.",

    "The name Adrastea is an alias for Lady Orvir; treat them as the same character.",

    "The dwarves of Doragon are connected to Madame Alora Venyette, not to Lady Orvir; \
     do not attribute this connection to Lady Orvir.",

    "Taelreth is the Institute of the Arcane in Diondria — it is not Lady Orvir's institution. \
     Lady Orvir heads the Sylvanian Academy of Magical Arts in Handletare.",

    "Some passages contain scripted read-aloud text addressed directly to players using \
     the word \"you\". Never reproduce this text verbatim and never address the reader as \
     \"you\". Convert any second-person language into third-person factual narration about \
     the character.",

    "Florian Reiter was a vampire for most of the earlier era but was restored to human \
     at the end of it. When describing him in the context of the earlier era, his vampiric \
     nature was central. In later eras he is human.",

    "Do not reference real-world proper nouns — countries, nationalities, ethnicities, \
     institutions, place names, or cultural labels from the real world (e.g. Romanian, \
     Jamaican, West Point, French, Eastern European). These appear in source passages only \
     as flavor notes for the author. Describe the trait or quality directly without the \
     real-world label, or omit it if it adds nothing.",

    "In this world, vampires can move around in daylight without significant difficulty as long \
     as they take basic precautions. Do not treat a vampire's ability to walk in daylight as \
     a notable fact, a special achievement, or a point of emphasis. It is not unusual.",
];

pub struct RagConfig {
    pub scene_markers: Vec<String>,
    pub prompt_extra_rules: Vec<String>,
}

impl RagConfig {
    /// Loads from compiled-in defaults. Either env var can be set to a pipe-separated
    /// list to override the entire default set for a specific deployment without a rebuild.
    pub fn load() -> Self {
        Self {
            scene_markers: env_override("SCENE_FILTER_MARKERS").unwrap_or_else(|| {
                DEFAULT_SCENE_MARKERS.iter().map(|s| s.to_lowercase()).collect()
            }),
            prompt_extra_rules: env_override("PROMPT_EXTRA_RULES").unwrap_or_else(|| {
                DEFAULT_PROMPT_RULES.iter().map(|s| s.to_string()).collect()
            }),
        }
    }
}

fn env_override(key: &str) -> Option<Vec<String>> {
    let val = std::env::var(key).ok()?;
    let items: Vec<String> = val
        .split('|')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if items.is_empty() { None } else { Some(items) }
}
