# Argus: pending work (as of 26 Oct 2026)

Everything up to here is committed on branch `week-9-v2-fixes-and-benchmark` and pushed. Nothing is running in AWS.
The new account's EC2 limits are 0 (blocked by AWS on 25 Oct).

## 1. AWS accounts

- [ ] **Old account 844641713781 (suspended):** finish the reactivation case with the card statement and ID. This comes first: other accounts are probably being linked to this one.
- [ ] **New account 906615256166 (EC2 blocked, limits 0):** check its email and Support Center for a review notice, and reply. Send one open message covering both accounts and the research.
- [ ] **Four quota cases on the new account** (32 vCPU Spot): 179292072900909 (eu-north-1), 179292073200039 (us-east-1), 179292073200709 (us-east-2), 179292055600697 (ap-southeast-1).
- [ ] **Do not create more AWS accounts.**
- [ ] **Alternatives if AWS does not restore access:**
  - apply for AWS Cloud Credit for Research;
  - ask Dr. Naresh Vurukonda about an NMIMS department or AWS Academy account;
  - use Angshuman's account, only if it is genuinely his (his card, his ID, his laptop).

## 2. Security cleanup (after the work is finished)

- [ ] Delete the new account's **root access key** created on 25 Oct (it was shared in a chat). Turn on root MFA.
- [ ] Delete the IAM user `argus-sensor` and its key: `scripts/setup_sensor_user.sh --delete`.
- [ ] Remove the runner role and the warning recorders: `scripts/setup_cloud_runner.sh --delete` and `scripts/setup_spot_signals.sh --delete`.
- [ ] Decide on the S3 bucket `argus-checkpoints-906615256166`. It holds a copy of the campaign data, which is also in git.
- [ ] Keep or remove the $130 budget `argus-campaign-spend`.
- [ ] Old account, if reactivated: delete its `argus-sensor` user, runner role and signal rules.

## 3. Automation still on the laptop

- [ ] The scheduled task **`argus-campaign-watchdog`** runs every 10 minutes until about 30 Oct.
  - It watches the Spot quotas and plays a notice if AWS restores them.
  - On **30 Oct, 15:30 IST**, its finish job rebuilds the results and paper numbers with the 160 reclaims, then commits and pushes. It resumes after a restart.
- [ ] To stop it early, in PowerShell: `Unregister-ScheduledTask -TaskName argus-campaign-watchdog`.

## 4. Paper (`Paper/v2`)

- [ ] Add the 6 new reclaims (152 to 158 sensor reclaims, 160 with the 2 earlier CloudTrail ones). Rerun the evaluation, capacity model, numbers, tables and figures; the finish job on 30 Oct does this.
- [ ] Rewrite the sentences that say "two days, four regions, 152".
- [ ] Fix the arXiv wording: the EKS drill pod was "rescheduled" by the drill script, not the operator. The operator now recreates it itself.
- [ ] Both authors read the whole paper before resubmission.
- [ ] Decide where to resubmit, and whether to post an arXiv v2.

## 5. Report and project page

- [ ] **Word report** `presentation_files/Argus_Project_Report.docx` is **local only** (`presentation_files/` is gitignored). Back it up, or force-add it to git.
- [ ] Update the report's numbers after the paper update (it currently says 152 reclaims).
- [ ] Update the project page in `site/`, then republish with `git subtree push --prefix site origin gh-pages`.
  Live at https://angshuman1211.github.io/Argus-Spot_Resilient_ML_Training_Orchestrator/

## 6. Code and repository

- [ ] **CI deploy role:** run `scripts/setup_github_oidc.sh` (needs a working AWS account), then set the repository variable `AWS_ACCOUNT_ID`.
- [ ] **Terraform:** the state still lists the deleted `argus-eks` cluster. `terraform/tfplan` and `terraform/zz_imports.tf` are uncommitted leftovers from another session; review them, then delete or commit.
- [ ] **Future work, if time allows:**
  - retrain the capacity model with more real labels;
  - make the operator move all nodes of a multi-node synchronous job together;
  - add a hybrid policy: periodic checkpoints plus alarms.

## 7. If AWS restores access and more data is wanted

- [ ] Restart slowly: 1 to 2 regions, a few sensors, no quota requests in the first days. Use the IAM user, not root keys.
- [ ] Edit `ml/data/real/campaign_state.json`, then run `python scripts/start_cloud_runner.py`. The spending stop (`stop_at_usd`) and the stop-on-block protection are built in.

## Key facts

- **Data:** 160 real reclaims (2 from CloudTrail, 152 sensors on 13 and 14 Oct, 6 sensors on 25 Oct), all in `ml/data/real/`, committed.
- **Main result:** the price-history model has no skill on real reclaims (AUROC 0.53). AWS's placement score reaches 0.74; placement plus instance age, 0.74 to 0.77.
- **Links:**
  - Paper: https://arxiv.org/abs/2609.39067
  - Code: https://github.com/Angshuman1211/Argus-Spot_Resilient_ML_Training_Orchestrator
  - Project page: https://angshuman1211.github.io/Argus-Spot_Resilient_ML_Training_Orchestrator/
