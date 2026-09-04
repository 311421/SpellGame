using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using System.IO;
using UnityEditor;
#endif

namespace SpellDrawing
{
    /// <summary>Editor workflow: draw a gesture in Play Mode, then use the context menu to save it as
    /// a SpellTemplate asset. Not needed at runtime in a shipped build.</summary>
    [RequireComponent(typeof(StrokeDrawer))]
    public class GestureTemplateRecorder : MonoBehaviour
    {
        [Tooltip("Folder (relative to the project) new template assets are saved into.")]
        [SerializeField] private string saveFolder = "Assets/_Project/Spells/Templates";

        [Tooltip("Name given to the next saved template.")]
        [SerializeField] private string nextSpellName = "New Spell";

        private StrokeDrawer _drawer;
        private List<List<Vector2>> _lastGesture;

        private void Awake()
        {
            _drawer = GetComponent<StrokeDrawer>();
            _drawer.OnGestureCompleted += gesture => _lastGesture = gesture;
        }

#if UNITY_EDITOR
        [ContextMenu("Save Last Gesture As Template")]
        private void SaveLastGesture()
        {
            if (_lastGesture == null || _lastGesture.Count == 0)
            {
                Debug.LogWarning("GestureTemplateRecorder: no gesture recorded yet — draw one and press the " +
                                  "cast key (Space by default) in Play Mode first.");
                return;
            }

            if (!Directory.Exists(saveFolder)) Directory.CreateDirectory(saveFolder);

            var template = ScriptableObject.CreateInstance<SpellTemplate>();
            template.spellName = nextSpellName;
            template.SetPointsFromRawStroke(StrokeDrawer.Flatten(_lastGesture));

            string path = AssetDatabase.GenerateUniqueAssetPath($"{saveFolder}/{nextSpellName}.asset");
            AssetDatabase.CreateAsset(template, path);
            AssetDatabase.SaveAssets();

            Debug.Log($"GestureTemplateRecorder: saved '{nextSpellName}' to {path}", template);
            EditorGUIUtility.PingObject(template);
        }
#endif
    }
}
