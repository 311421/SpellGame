using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace SpellDrawing
{
    /// <summary>A single recognizable spell gesture: shape data used for matching, plus what happens
    /// when a player draws it. Create via GestureTemplateRecorder or Assets &gt; Create &gt; Spell Drawing.</summary>
    [CreateAssetMenu(fileName = "New Spell Template", menuName = "Spell Drawing/Spell Template")]
    public class SpellTemplate : ScriptableObject
    {
        [Tooltip("Display name for this spell.")]
        public string spellName = "New Spell";

        [Tooltip("Normalized gesture points used for $1 matching. Auto-filled — don't hand-edit.")]
        [SerializeField] private List<Vector2> normalizedPoints = new List<Vector2>();

        [Tooltip("Raw drawn stroke, never derotated. What SyntheticDatasetGenerator augments from. " +
                 "Auto-filled — don't hand-edit. Empty on templates recorded before this field existed.")]
        [SerializeField] private List<Vector2> capturedPoints = new List<Vector2>();

        [Tooltip("0 (round) to 1 (sharp corners). Auto-filled — don't hand-edit. Use 'Recompute Corner " +
                 "Score' below if this predates the feature.")]
        [SerializeField] private float cornerScore;

        [Header("Cast Result")]
        [Tooltip("Spawned at the caster's origin when this spell is successfully cast. Optional.")]
        public GameObject effectPrefab;

        [Tooltip("Minimum recognizer confidence (0-1) required to cast this spell. Raise this if it's " +
                 "being confused with other shapes; lower it if it's too hard to trigger.")]
        [Range(0f, 1f)] public float minConfidence = 0.75f;

        [Tooltip("Seconds before this spell can be cast again after a successful cast.")]
        public float cooldown = 0.5f;

        [Header("CV / ML Training")]
        [Tooltip("Rotation range (+/-deg) SyntheticDatasetGenerator randomizes for this spell. Wide " +
                 "(up to 180) for orientation-agnostic shapes; narrow (~20-30) for orientation-sensitive ones.")]
        [Range(0f, 180f)] public float maxTrainingRotationDeg = 30f;

        public IReadOnlyList<Vector2> NormalizedPoints => normalizedPoints;
        public IReadOnlyList<Vector2> CapturedPoints => capturedPoints;
        public float CornerScore => cornerScore;

        /// <summary>Stores the raw drawn stroke as-is, then runs it through the $1 normalization
        /// pipeline for matching, along with its corner score.</summary>
        public void SetPointsFromRawStroke(IReadOnlyList<Vector2> rawPoints)
        {
            capturedPoints = new List<Vector2>(rawPoints);
            normalizedPoints = DollarOneRecognizer.Normalize(rawPoints);
            cornerScore = DollarOneRecognizer.ComputeCornerScore(normalizedPoints);
        }

#if UNITY_EDITOR
        [ContextMenu("Recompute Corner Score")]
        private void RecomputeCornerScore()
        {
            cornerScore = DollarOneRecognizer.ComputeCornerScore(normalizedPoints);
            EditorUtility.SetDirty(this);
            Debug.Log($"'{spellName}' corner score recomputed: {cornerScore:F2}", this);
        }
#endif
    }
}
