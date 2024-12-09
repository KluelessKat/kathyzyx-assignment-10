document.querySelector('input[type="file"]').addEventListener('change', function(e) {
    const filename = e.target.files[0] ? e.target.files[0].name : '';
    document.getElementById('selected-file').textContent = filename;
});

document.querySelector('select[name="query_type"]').addEventListener('change', function(e) {
    const weight = document.querySelector('input[name="weight"]');
    weight.disabled = e.target.value !== 'hybrid';
});
