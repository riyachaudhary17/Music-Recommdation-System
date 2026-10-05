const apiBase = 'http://127.0.0.1:8000';
const movieMetaMap = new Map();

const form = document.getElementById('recommendForm');
const userIdInput = document.getElementById('userId');
const methodSelect = document.getElementById('method');
const topNInput = document.getElementById('topN');
const searchInput = document.getElementById('searchInput');
const resultsEl = document.getElementById('results');
const statusEl = document.getElementById('status');
const loadingEl = document.getElementById('loading');
const errorEl = document.getElementById('errorState');

const parseCsvLine = (line) => {
  const values = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < line.length; i += 1) {
    const ch = line[i];
    const next = line[i + 1];

    if (ch === '"') {
      if (inQuotes && next === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
      continue;
    }

    if (ch === ',' && !inQuotes) {
      values.push(current);
      current = '';
      continue;
    }

    current += ch;
  }

  values.push(current);
  return values.map((value) => value.trim());
};

async function loadMovieMetadata() {
  try {
    const response = await fetch('/data/processed/movies.csv');
    if (!response.ok) {
      throw new Error('Could not load local movie catalog metadata.');
    }

    const csvText = await response.text();
    const lines = csvText.split(/\r?\n/).filter(Boolean);
    if (lines.length < 2) {
      return;
    }

    const headers = parseCsvLine(lines[0]);
    for (let i = 1; i < lines.length; i += 1) {
      const values = parseCsvLine(lines[i]);
      const row = {};
      headers.forEach((header, index) => {
        row[header] = values[index] || '';
      });

      if (!row.movieId) continue;
      movieMetaMap.set(Number(row.movieId), {
        title: row.title || '',
        genres: row.genres || 'Unknown genre',
      });
    }
  } catch (error) {
    console.warn('Movie metadata not loaded:', error);
  }
}

function setLoading(isLoading) {
  loadingEl.classList.toggle('hidden', !isLoading);
  if (!isLoading) {
    statusEl.textContent = 'Recommendations ready';
  }
}

function showError(message) {
  errorEl.textContent = message;
  errorEl.classList.remove('hidden');
}

function clearError() {
  errorEl.textContent = '';
  errorEl.classList.add('hidden');
}

function getPosterInitials(title) {
  const parts = title.split(/\s+/).filter(Boolean).slice(0, 2);
  if (!parts.length) return 'MV';
  return parts.map((part) => part[0].toUpperCase()).join('').slice(0, 2);
}

function getSentimentClass(label) {
  if (!label) return 'sentiment-neutral';
  if (label === 'positive') return 'sentiment-positive';
  if (label === 'negative') return 'sentiment-negative';
  return 'sentiment-neutral';
}

function buildCardMarkup(movie) {
  const movieId = Number(movie.movieId);
  const title = movie.title || 'Unknown title';
  const meta = movieMetaMap.get(movieId);
  const genreText = meta && meta.genres ? meta.genres.replace(/\|/g, ', ') : 'Genres unavailable';
  const scoreValue = typeof movie.score === 'number' ? movie.score : 0;
  const ratingText = `${(scoreValue * 10).toFixed(1)}/10`;
  const sentiment = movie.sentiment || null;
  const sentimentLabel = sentiment?.label || 'neutral';
  const sentimentCompound = sentiment?.scores?.compound ?? 0;
  const overview = `A highly recommended title from the local catalog. ${genreText} and a ${sentimentLabel} audience response.`;

  return `
    <article class="movie-card">
      <div class="poster" aria-label="${title} poster placeholder">
        <span>${getPosterInitials(title)}</span>
      </div>
      <div class="card-body">
        <div class="movie-header">
          <h3>${title}</h3>
          <span class="score-pill">${ratingText}</span>
        </div>
        <p class="movie-meta">Genres: ${genreText}</p>
        <p class="overview">${overview}</p>
        <div class="sentiment-row">
          <span class="sentiment-badge ${getSentimentClass(sentimentLabel)}">${sentimentLabel}</span>
          <span class="sentiment-score">compound ${sentimentCompound.toFixed(2)}</span>
        </div>
      </div>
    </article>
  `;
}

function renderMovies(movies) {
  const searchTerm = searchInput.value.trim().toLowerCase();
  const filtered = movies.filter((movie) => {
    if (!searchTerm) return true;
    return (movie.title || '').toLowerCase().includes(searchTerm);
  });

  if (!filtered.length) {
    resultsEl.innerHTML = '<div class="empty-state">No movies match the current search or recommendation results.</div>';
    return;
  }

  resultsEl.innerHTML = filtered.map(buildCardMarkup).join('');
}

async function loadRecommendations(event) {
  if (event) {
    event.preventDefault();
  }

  clearError();
  setLoading(true);

  const payload = {
    user_id: Number(userIdInput.value),
    method: methodSelect.value,
    top_n: Number(topNInput.value),
  };

  try {
    const response = await fetch(`${apiBase}/recommend`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorPayload = await response.json().catch(() => ({}));
      throw new Error(errorPayload.detail || 'Unable to get recommendations.');
    }

    const data = await response.json();
    statusEl.textContent = `Showing recommendations for user ${payload.user_id} (${payload.method})`;
    renderMovies(data.results || []);
  } catch (error) {
    showError(error.message || 'Something went wrong while loading recommendations.');
    resultsEl.innerHTML = '';
  } finally {
    setLoading(false);
  }
}

searchInput.addEventListener('input', () => {
  const currentResults = resultsEl.querySelectorAll('.movie-card');
  if (!currentResults.length) {
    return;
  }

  const filterValue = searchInput.value.trim().toLowerCase();
  const cards = Array.from(currentResults);
  cards.forEach((card) => {
    const title = card.querySelector('h3')?.textContent?.toLowerCase() || '';
    card.style.display = title.includes(filterValue) || !filterValue ? 'flex' : 'none';
  });
});

form.addEventListener('submit', loadRecommendations);

window.addEventListener('DOMContentLoaded', async () => {
  await loadMovieMetadata();
  loadRecommendations();
});
