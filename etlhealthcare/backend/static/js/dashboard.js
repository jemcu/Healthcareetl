// Dashboard charts — Chart.js
(function () {
  const palette = {
    ink: '#11201E', teal: '#0F4C46', tealLight: '#3E8A82',
    gold: '#C99B5B', rust: '#B5523A', moss: '#5C7A4E',
    amber: '#D78A2B', cream: '#F4EFE2', line: '#E4DFD3'
  };
  const riskColors = { bajo: palette.moss, medio: palette.amber, alto: palette.rust, critico: palette.ink };
  const J = (id) => JSON.parse(document.getElementById(id).textContent);

  Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
  Chart.defaults.color = '#5A6B68';
  Chart.defaults.borderColor = palette.line;

  // Opciones base compartidas
  const baseOpts = {
    maintainAspectRatio: false,
    responsive: true,
    layout: { padding: { top: 4, bottom: 4 } },
    plugins: {
      legend: {
        labels: { boxWidth: 10, font: { size: 11 } }
      }
    }
  };

  // Riesgo (doughnut)
  const riesgo = J('data-riesgo');
  new Chart(document.getElementById('chartRiesgo'), {
    type: 'doughnut',
    data: {
      labels: riesgo.map(r => r.riesgo_enfermedad),
      datasets: [{
        data: riesgo.map(r => r.total),
        backgroundColor: riesgo.map(r => riskColors[r.riesgo_enfermedad] || palette.teal),
        borderWidth: 0
      }]
    },
    options: {
      ...baseOpts,
      cutout: '62%',
      plugins: {
        ...baseOpts.plugins,
        legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 11 } } }
      }
    }
  });

  // Edad (line)
  const edad = J('data-edad');
  new Chart(document.getElementById('chartEdad'), {
    type: 'line',
    data: {
      labels: edad.map(e => e.rango),
      datasets: [
        { label: 'Total',    data: edad.map(e => e.total),    borderColor: palette.teal,  backgroundColor: 'rgba(15,76,70,.12)', tension: .35, fill: true, borderWidth: 2.5 },
        { label: 'Alto',     data: edad.map(e => e.alto),     borderColor: palette.amber, tension: .35, borderWidth: 2 },
        { label: 'Críticos', data: edad.map(e => e.criticos), borderColor: palette.rust,  tension: .35, borderWidth: 2 }
      ]
    },
    options: {
      ...baseOpts,
      plugins: {
        ...baseOpts.plugins,
        legend: { position: 'bottom', labels: { boxWidth: 10, font: { size: 11 } } }
      },
      scales: {
        y: { beginAtZero: true, grid: { color: palette.line } },
        x: { grid: { display: false } }
      }
    }
  });

  // Sexo (bar)
  const sexo = J('data-sexo');
  const sexoLabel = { M: 'Masculino', F: 'Femenino', O: 'Otro' };
  new Chart(document.getElementById('chartSexo'), {
    type: 'bar',
    data: {
      labels: sexo.map(s => sexoLabel[s.sexo] || s.sexo),
      datasets: [{ data: sexo.map(s => s.total), backgroundColor: [palette.teal, palette.gold, palette.moss], borderRadius: 8 }]
    },
    options: {
      ...baseOpts,
      plugins: {
        ...baseOpts.plugins,
        legend: { display: false }
      },
      scales: {
        y: { grid: { color: palette.line } },
        x: { grid: { display: false } }
      }
    }
  });

  // IMC (bar)
  const imc = J('data-imc');
  const imcLabel = { bajo_peso: 'Bajo peso', normal: 'Normal', sobrepeso: 'Sobrepeso', obesidad: 'Obesidad' };
  new Chart(document.getElementById('chartIMC'), {
    type: 'bar',
    data: {
      labels: imc.map(i => imcLabel[i.imc_clasificacion]),
      datasets: [{ data: imc.map(i => i.total), backgroundColor: palette.teal, borderRadius: 8 }]
    },
    options: {
      ...baseOpts,
      plugins: {
        ...baseOpts.plugins,
        legend: { display: false }
      },
      scales: {
        y: { grid: { color: palette.line } },
        x: { grid: { display: false } }
      }
    }
  });

  // Diagnósticos (horizontal bar)
  const diag = J('data-diag');
  new Chart(document.getElementById('chartDiag'), {
    type: 'bar',
    data: {
      labels: diag.map(d => d.diagnostico_preliminar),
      datasets: [{ data: diag.map(d => d.total), backgroundColor: palette.gold, borderRadius: 6 }]
    },
    options: {
      ...baseOpts,
      indexAxis: 'y',
      plugins: {
        ...baseOpts.plugins,
        legend: { display: false }
      },
      scales: {
        x: { grid: { color: palette.line } },
        y: { grid: { display: false } }
      }
    }
  });

})();

function fillLogin(user, pass) {
  const u = document.getElementById('id_username') 
         || document.querySelector('input[name="username"]');
  const p = document.getElementById('id_password') 
         || document.querySelector('input[name="password"]');
  if (u) { u.value = user; u.dispatchEvent(new Event('input', { bubbles: true })); }
  if (p) { p.value = pass; p.dispatchEvent(new Event('input', { bubbles: true })); }
}
