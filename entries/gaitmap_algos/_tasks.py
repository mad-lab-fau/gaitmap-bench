from pathlib import Path
import git

HERE = Path(__file__).parent
MAIN_GIT_REPO = HERE.parent.parent


def commit_lock_if_changed():
    repo = git.Repo(MAIN_GIT_REPO)

    # Check if poetry.lock has changed
    if not repo.is_dirty(untracked_files=True):
        print("Repo is clean. Nothing to commit.")
        return

    poetry_lock = (HERE / "poetry.lock").relative_to(MAIN_GIT_REPO)
    here_relative = HERE.relative_to(MAIN_GIT_REPO)
    all_changed_files = [item.a_path for item in repo.index.diff(None)]
    if str(poetry_lock) not in all_changed_files:
        print("Poetry.lock has not changed. Nothing to commit.")
        return
    if any(str(item).startswith(str(here_relative)) for item in all_changed_files):
        print(f"Poetry.lock has changed, but other files of the entry ({here_relative}) have changed too. "
              "Assuming debug run. Not committing.")
        return
    print("Poetry.lock has changed. Committing.")
    if len(repo.index.diff("HEAD")) > 0:
        raise RuntimeError("There are staged changes in the repo. Can not commit changed poetry.lock cleanly.")
    repo.git.add(poetry_lock)
    repo.index.commit("Update poetry.lock after run update.")
