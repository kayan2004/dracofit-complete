import React, { useState, useEffect, useRef } from "react";
import ExerciseCard from "./ExerciseCard";
import ExerciseFilters from "./ExerciseFilters"; // Update the import
import exercisesService from "../../services/exercisesService";

/**
 * Modal component for selecting exercises
 * @param {boolean} isOpen - Whether the modal is open
 * @param {function} onClose - Function to call when the modal is closed
 * @param {function} onExerciseSelect - Function to call when an exercise is selected with params (exerciseId, exerciseInfo)
 * @param {Object} initialParams - Optional initial parameters for sets, reps, rest time
 */
const ExercisesModal = ({
  isOpen,
  onClose,
  onExerciseSelect,
  initialParams = {},
}) => {
  const [exercises, setExercises] = useState([]);
  const [filteredExercises, setFilteredExercises] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [showFilters, setShowFilters] = useState(false);

  // Updated to use the enhanced filters structure
  const [filters, setFilters] = useState({
    difficulty: "",
    targetMuscles: [],
    equipment: "",
    type: "",
  });

  const [selectedExerciseId, setSelectedExerciseId] = useState("");
  const [exerciseParams, setExerciseParams] = useState({
    sets: initialParams.sets || 3,
    reps: initialParams.reps || 10,
    restTimeSeconds: initialParams.restTimeSeconds || 60,
  });

  const searchInputRef = useRef(null);
  const modalRef = useRef(null);

  // Focus search input when modal opens
  useEffect(() => {
    if (isOpen && searchInputRef.current) {
      setTimeout(() => {
        searchInputRef.current.focus();
      }, 100);
    }
  }, [isOpen]);

  // Fetch all exercises when modal opens
  useEffect(() => {
    const fetchExercises = async () => {
      if (!isOpen) return;

      try {
        setLoading(true);
        setError(null);

        // Use the existing getFilteredExercises method with a large limit
        const response = await exercisesService.getFilteredExercises(
          {},
          1,
          1000
        );

        // Handle different response formats
        let fetchedExercises = [];
        if (response.exercises && Array.isArray(response.exercises)) {
          fetchedExercises = response.exercises;
        } else if (Array.isArray(response)) {
          fetchedExercises = response;
        }

        setExercises(fetchedExercises);
        setFilteredExercises(fetchedExercises);
      } catch (err) {
        console.error("Error fetching exercises:", err);
        setError(err.message || "Failed to load exercises");
      } finally {
        setLoading(false);
      }
    };

    fetchExercises();
  }, [isOpen]);

  // Apply filters when search term or filters change
  useEffect(() => {
    if (!exercises.length) return;

    const applyFilters = async () => {
      // Check if we have any active filters
      const hasActiveFilters =
        searchTerm ||
        filters.difficulty ||
        filters.equipment ||
        filters.type ||
        filters.targetMuscles.length > 0;

      // If no filters are applied, show all exercises
      if (!hasActiveFilters) {
        setFilteredExercises(exercises);
        return;
      }

      try {
        setLoading(true);

        // Create filter object for API call
        const apiFilters = { ...filters };

        // Add search if provided
        if (searchTerm) {
          apiFilters.search = searchTerm;
        }

        // Call API with all filters
        const filterResponse = await exercisesService.getFilteredExercises(
          apiFilters,
          1,
          1000
        );

        let filterResults = [];
        if (
          filterResponse.exercises &&
          Array.isArray(filterResponse.exercises)
        ) {
          filterResults = filterResponse.exercises;
        } else if (Array.isArray(filterResponse)) {
          filterResults = filterResponse;
        }

        setFilteredExercises(filterResults);
      } catch (err) {
        console.error("Error applying filters:", err);

        // Fall back to client-side filtering if the server request fails
        const filtered = exercises.filter((exercise) => {
          // Text search matching
          const matchesSearch =
            !searchTerm.trim() ||
            exercise.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
            (exercise.description &&
              exercise.description
                .toLowerCase()
                .includes(searchTerm.toLowerCase())) ||
            (exercise.targetMuscles &&
              exercise.targetMuscles.some((muscle) =>
                muscle.toLowerCase().includes(searchTerm.toLowerCase())
              ));

          // Difficulty matching
          const matchesDifficulty =
            !filters.difficulty ||
            (exercise.difficulty &&
              exercise.difficulty.toLowerCase() ===
                filters.difficulty.toLowerCase());

          // Equipment matching
          const matchesEquipment =
            !filters.equipment ||
            (exercise.equipment &&
              exercise.equipment.toLowerCase() ===
                filters.equipment.toLowerCase());

          // Type matching
          const matchesType =
            !filters.type ||
            (exercise.type &&
              exercise.type.toLowerCase() === filters.type.toLowerCase());

          // Target muscles matching (any of the selected muscles)
          const matchesMuscles =
            filters.targetMuscles.length === 0 ||
            (exercise.targetMuscles &&
              exercise.targetMuscles.some((muscle) =>
                filters.targetMuscles.includes(muscle.toLowerCase())
              ));

          return (
            matchesSearch &&
            matchesDifficulty &&
            matchesEquipment &&
            matchesType &&
            matchesMuscles
          );
        });

        setFilteredExercises(filtered);
      } finally {
        setLoading(false);
      }
    };

    // Apply filters with a small debounce for better UX
    const timerId = setTimeout(() => {
      applyFilters();
    }, 300);

    return () => clearTimeout(timerId);
  }, [searchTerm, filters, exercises]);

  // Handle parameter changes (sets, reps, rest)
  const handleParamChange = (e) => {
    const { name, value } = e.target;
    const numValue = parseInt(value, 10);

    if (isNaN(numValue)) return;

    setExerciseParams((prev) => ({
      ...prev,
      [name]: numValue,
    }));
  };

  // Handle filter changes
  const handleFilterChange = (filterType, value) => {
    setFilters((prev) => {
      // Special handling for muscle groups (array)
      if (filterType === "targetMuscles") {
        // If value is already in array, remove it; otherwise add it
        const updatedMuscles = prev.targetMuscles.includes(value)
          ? prev.targetMuscles.filter((muscle) => muscle !== value)
          : [...prev.targetMuscles, value];

        return { ...prev, targetMuscles: updatedMuscles };
      }

      // For other filters (single value)
      return { ...prev, [filterType]: value };
    });
  };

  // Clear all filters
  const clearFilters = () => {
    setFilters({
      difficulty: "",
      targetMuscles: [],
      equipment: "",
      type: "",
    });
    setSearchTerm("");
  };

  // Handle outside click to close modal
  const handleOutsideClick = (e) => {
    if (modalRef.current && !modalRef.current.contains(e.target)) {
      onClose();
    }
  };

  // Handle add exercise
  const handleAddExercise = () => {
    if (!selectedExerciseId) return;

    const selectedExercise = exercises.find(
      (ex) =>
        ex.id === parseInt(selectedExerciseId, 10) ||
        ex.id === selectedExerciseId
    );

    if (selectedExercise && onExerciseSelect) {
      onExerciseSelect(selectedExercise, exerciseParams);
      onClose();
    }
  };

  // Don't render anything if modal is not open
  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 z-50 overflow-auto bg-black bg-opacity-75 flex items-center justify-center p-4"
      onClick={handleOutsideClick}
    >
      <div
        ref={modalRef}
        className="bg-dark-slate-gray border border-gray-700 rounded-lg w-full max-w-4xl max-h-[90vh] overflow-hidden flex flex-col animate-fadeIn"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="bg-gray-800 px-6 py-4 flex items-center justify-between border-b border-gray-700">
          <h3 className="text-xl font-bold text-goldenrod">Select Exercise</h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
            aria-label="Close modal"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth="2"
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Modal Body */}
        <div className="flex-1 overflow-auto p-6">
          {error && (
            <div className="bg-red-900/20 border border-red-500 text-white p-4 rounded-lg mb-6">
              <p>{error}</p>
              <button
                className="text-sm text-red-400 underline mt-2"
                onClick={() => setError(null)}
              >
                Dismiss
              </button>
            </div>
          )}

          {/* Search Input (Separate from filters for better UX) */}
          <div className="mb-4">
            <input
              ref={searchInputRef}
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Search exercises..."
              className="w-full bg-midnight-green border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-goldenrod"
            />
          </div>

          {/* Replace the old ExerciseFilter with ExerciseFilters */}
          <ExerciseFilters
            filters={filters}
            onFilterChange={handleFilterChange}
            onClearFilters={clearFilters}
            showFilters={showFilters}
            onToggleFilters={() => setShowFilters(!showFilters)}
          />

          {/* Exercise List */}
          <div className="mt-6">
            <div className="flex justify-between items-center mb-3">
              <h4 className="text-md font-medium text-goldenrod">
                Available Exercises
              </h4>
              <span className="text-sm text-gray-400">
                {filteredExercises.length} exercise
                {filteredExercises.length !== 1 ? "s" : ""}
              </span>
            </div>

            {loading ? (
              <div className="flex justify-center items-center py-10">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-goldenrod"></div>
              </div>
            ) : (
              <div className="bg-gray-800/50 rounded-lg p-2 max-h-[40vh] overflow-y-auto">
                {filteredExercises.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    {filteredExercises.map((exercise) => (
                      <div
                        key={exercise.id}
                        className={`rounded-lg cursor-pointer transition-all ${
                          selectedExerciseId === exercise.id
                            ? "ring-2 ring-goldenrod transform scale-[1.02]"
                            : "hover:bg-gray-700/50"
                        }`}
                        onClick={() => setSelectedExerciseId(exercise.id)}
                      >
                        <ExerciseCard
                          exercise={exercise}
                          compact={true}
                          selected={selectedExerciseId === exercise.id}
                          hideCTA={true}
                        />
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-center py-8 text-gray-400">
                    {searchTerm ||
                    Object.values(filters).some((v) =>
                      Array.isArray(v) ? v.length > 0 : Boolean(v)
                    )
                      ? "No exercises match your search criteria"
                      : "No exercises available"}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Exercise Details */}
          {selectedExerciseId && (
            <div className="mt-6 bg-gray-800 rounded-lg p-4">
              <h4 className="text-md font-medium text-goldenrod mb-4">
                Exercise Parameters
              </h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <label
                    htmlFor="sets"
                    className="block text-sm font-medium mb-1 text-gray-300"
                  >
                    Sets
                  </label>
                  <input
                    type="number"
                    id="sets"
                    name="sets"
                    min="1"
                    max="20"
                    value={exerciseParams.sets}
                    onChange={handleParamChange}
                    className="w-full bg-midnight-green border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-goldenrod"
                  />
                </div>
                <div>
                  <label
                    htmlFor="reps"
                    className="block text-sm font-medium mb-1 text-gray-300"
                  >
                    Reps
                  </label>
                  <input
                    type="number"
                    id="reps"
                    name="reps"
                    min="1"
                    max="100"
                    value={exerciseParams.reps}
                    onChange={handleParamChange}
                    className="w-full bg-midnight-green border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-goldenrod"
                  />
                </div>
                <div>
                  <label
                    htmlFor="restTimeSeconds"
                    className="block text-sm font-medium mb-1 text-gray-300"
                  >
                    Rest (seconds)
                  </label>
                  <input
                    type="number"
                    id="restTimeSeconds"
                    name="restTimeSeconds"
                    min="0"
                    max="300"
                    value={exerciseParams.restTimeSeconds}
                    onChange={handleParamChange}
                    className="w-full bg-midnight-green border border-gray-600 rounded-lg px-4 py-2 text-white focus:outline-none focus:border-goldenrod"
                  />
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Modal Footer */}
        <div className="bg-gray-800 px-6 py-4 flex justify-between border-t border-gray-700">
          <button
            type="button"
            onClick={onClose}
            className="px-4 py-2 bg-gray-700 text-white rounded-lg hover:bg-gray-600 transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={handleAddExercise}
            disabled={!selectedExerciseId}
            className={`px-6 py-2 rounded-lg font-medium ${
              !selectedExerciseId
                ? "bg-gray-600 text-gray cursor-not-allowed"
                : "bg-goldenrod text-midnight-green hover:bg-dark-goldenrod"
            }`}
          >
            Add to Workout
          </button>
        </div>
      </div>
    </div>
  );
};

export default ExercisesModal;
