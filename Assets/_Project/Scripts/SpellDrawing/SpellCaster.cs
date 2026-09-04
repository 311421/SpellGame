using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Events;

namespace SpellDrawing
{
    [System.Serializable]
    public class SpellCastEvent : UnityEvent<SpellTemplate> { }

    /// <summary>Matches a StrokeDrawer's completed gestures against SpellTemplates and casts
    /// (spawns effectPrefab + fires OnSpellCast) on a confident, non-ambiguous, off-cooldown match.</summary>
    [RequireComponent(typeof(StrokeDrawer))]
    public class SpellCaster : MonoBehaviour
    {
        [Tooltip("The set of gestures this caster can recognize.")]
        [SerializeField] private List<SpellTemplate> spells = new List<SpellTemplate>();

        [Tooltip("Where cast effects are spawned. Defaults to this GameObject's transform.")]
        [SerializeField] private Transform castOrigin;

        [Header("Events")]
        public SpellCastEvent OnSpellCast;
        public UnityEvent OnRecognitionFailed;

        [Header("Debug")]
        [Tooltip("Logs stroke point count and match results to the Console. Turn off before shipping, or " +
                 "wrap with a build define if you want it gone from release builds entirely.")]
        [SerializeField] private bool logMatching = true;

        [Tooltip("How many of the top-scoring candidate spells to include in the log line. " +
                 "Handy for spotting shapes that are consistently confused with each other.")]
        [SerializeField] private int logTopCandidates = 3;

        [Header("Ambiguity Guard")]
        [Tooltip("Winning spell must beat the runner-up's score by at least this much, even if it clears " +
                 "its own minConfidence — guards against near-ties between structurally similar shapes.")]
        [SerializeField, Range(0f, 0.5f)] private float minScoreMargin = 0.08f;

        private StrokeDrawer _drawer;
        private readonly Dictionary<SpellTemplate, float> _cooldownUntil = new Dictionary<SpellTemplate, float>();

        private void Awake()
        {
            _drawer = GetComponent<StrokeDrawer>();
            if (castOrigin == null) castOrigin = transform;
        }

        private void OnEnable() => _drawer.OnGestureCompleted += HandleGestureCompleted;
        private void OnDisable() => _drawer.OnGestureCompleted -= HandleGestureCompleted;

        private void HandleGestureCompleted(List<List<Vector2>> strokes)
        {
            List<Vector2> rawPoints = StrokeDrawer.Flatten(strokes);
            DollarOneRecognizer.Result result = DollarOneRecognizer.Recognize(rawPoints, spells);

            if (logMatching) LogCandidates(rawPoints.Count, result);

            if (result.template == null || result.score < result.template.minConfidence)
            {
                if (logMatching) Debug.Log("[SpellCaster] No spell cleared its confidence threshold — treating as a miss.", this);
                OnRecognitionFailed?.Invoke();
                return;
            }

            if (result.ranked.Count > 1)
            {
                float margin = result.ranked[0].score - result.ranked[1].score;
                if (margin < minScoreMargin)
                {
                    if (logMatching)
                    {
                        Debug.Log($"[SpellCaster] '{result.ranked[0].template.spellName}' ({result.ranked[0].score:F2}) " +
                                  $"too close to '{result.ranked[1].template.spellName}' ({result.ranked[1].score:F2}) " +
                                  $"— margin {margin:F2} < {minScoreMargin:F2}. Rejecting as ambiguous.", this);
                    }
                    OnRecognitionFailed?.Invoke();
                    return;
                }
            }

            if (_cooldownUntil.TryGetValue(result.template, out float readyTime) && Time.time < readyTime)
            {
                if (logMatching)
                {
                    Debug.Log($"[SpellCaster] '{result.template.spellName}' matched (score {result.score:F2}) " +
                              $"but is on cooldown for {readyTime - Time.time:F2}s more — ignored.", this);
                }
                return;
            }

            _cooldownUntil[result.template] = Time.time + result.template.cooldown;
            if (logMatching) Debug.Log($"[SpellCaster] Casting '{result.template.spellName}' (score {result.score:F2}).", this);
            Cast(result.template);
        }

        private void LogCandidates(int rawPointCount, DollarOneRecognizer.Result result)
        {
            if (result.ranked == null || result.ranked.Count == 0)
            {
                Debug.Log($"[SpellCaster] Stroke captured ({rawPointCount} raw points) — no spell templates assigned.", this);
                return;
            }

            var sb = new StringBuilder();
            sb.Append($"[SpellCaster] Stroke captured ({rawPointCount} raw points) -> ");

            int shown = Mathf.Min(logTopCandidates, result.ranked.Count);
            for (int i = 0; i < shown; i++)
            {
                (SpellTemplate template, float score) = result.ranked[i];
                sb.Append($"{template.spellName}: {score:F2} (needs {template.minConfidence:F2})");
                if (i < shown - 1) sb.Append(", ");
            }

            Debug.Log(sb.ToString(), this);
        }

        private void Cast(SpellTemplate spell)
        {
            if (spell.effectPrefab != null)
            {
                Instantiate(spell.effectPrefab, castOrigin.position, castOrigin.rotation);
            }
            OnSpellCast?.Invoke(spell);
        }
    }
}
