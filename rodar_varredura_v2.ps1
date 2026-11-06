$ErrorActionPreference = 'Stop'
$env:HF_HUB_OFFLINE = '1'
$common = @(
    '--model','eduagarcia/RoBERTaLexPT-base',
    '--revision','9dea30875c93dc2d33b35c2338dfc93c86715b9e',
    '--mode','comprehensive','--seed','42','--epochs','12',
    '--selection','iou50','--scheduler','linear','--patience','12',
    '--batch-size','2','--skip-test','--include-partial','--auxiliary',
    '--init-checkpoint','artefatos/compare_v2_robertalexpt_base_s42/best.pt',
    '--source-jsonl','artefatos/enrichment_v2/versoes_pronunciamentos.jsonl'
)
$runs = @(
    @{Name='lr5e6_i2'; Rate='5e-6'; Inside='2'},
    @{Name='lr2e5_i2'; Rate='2e-5'; Inside='2'},
    @{Name='lr2e5_i1'; Rate='2e-5'; Inside='1'},
    @{Name='lr2e5_i3'; Rate='2e-5'; Inside='3'}
)
foreach ($run in $runs) {
    Write-Output "START $($run.Name)"
    python -u train.py @common --learning-rate $run.Rate --inside-weight $run.Inside --run-dir "artefatos/sweep_v2_$($run.Name)_s42"
    if ($LASTEXITCODE -ne 0) { throw "Falhou $($run.Name) com código $LASTEXITCODE" }
    Write-Output "DONE $($run.Name)"
}
