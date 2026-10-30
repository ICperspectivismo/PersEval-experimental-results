"""Local Git preflight and allowlisted commits. No transports or history rewrites."""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .lfs import eligible, matches, pointer_bytes, LFS_THRESHOLD
from .protocol import ProtocolError


def git_process(root, *args, private_index=None):
    # Inherited overrides must never redirect a publication into a different repository.
    env = {key:value for key,value in os.environ.items() if key not in {
        'GIT_DIR','GIT_WORK_TREE','GIT_INDEX_FILE','GIT_OBJECT_DIRECTORY',
        'GIT_ALTERNATE_OBJECT_DIRECTORIES','GIT_COMMON_DIR','GIT_NAMESPACE',
    }}
    env.update(GIT_TERMINAL_PROMPT='0',GIT_OPTIONAL_LOCKS='0',GIT_NO_LAZY_FETCH='1',GIT_ALLOW_PROTOCOL='',GIT_LFS_SKIP_SMUDGE='1')
    if private_index is not None:
        env['GIT_INDEX_FILE'] = str(private_index)
    # check-ignore consumes literal filesystem paths and rejects pathspec magic.
    literal = [] if args and args[0]=='check-ignore' else ['--literal-pathspecs']
    return subprocess.run(['git',*literal,'-C',str(root),*args],capture_output=True,env=env)


def git(root, *args):
    proc = git_process(root,*args)
    if proc.returncode:
        raise ProtocolError(f'git {args[0]} failed: {proc.stderr.decode(errors="replace").strip()}')
    return proc.stdout.decode('utf-8').strip()


@dataclass
class GitState:
    head: str
    branch: str
    index: bytes | None
    config: bytes
    upstream: dict


@dataclass
class CommitOutcome:
    sha: str
    warnings: list[str]


class DurableCommitError(ProtocolError):
    """A commit exists. Never rollback its files or rewrite history."""
    def __init__(self, sha, message):
        self.sha = sha
        super().__init__(message)


class GitRepository:
    def __init__(self, root: Path):
        self.root = root
        if Path(git(root,'rev-parse','--show-toplevel')).resolve() != root:
            raise ProtocolError('--repo must select the results repository root')

    def path(self, name):
        path = Path(git(self.root,'rev-parse','--git-path',name))
        return path if path.is_absolute() else self.root/path

    def head(self):
        proc = git_process(self.root,'rev-parse','--verify','HEAD')
        return proc.stdout.decode().strip() if proc.returncode == 0 else None

    def head_files(self, head=None):
        head = head or self.head()
        if head is None:
            return {}
        names = git(self.root,'ls-tree','-r','--name-only',head,'--','runs','registry').splitlines()
        result = {}
        for name in names:
            proc = git_process(self.root,'show',f'{head}:{name}')
            if proc.returncode:
                raise ProtocolError(f'cannot read committed source: {name}')
            result[name] = proc.stdout
        return result

    def ensure_immutable(self, working):
        for name,data in self.head_files().items():
            if name.startswith('registry/submissions/') and name.endswith('/current.json'):
                if name not in working:
                    raise ProtocolError(f'committed current pointer was deleted: {name}')
            elif name not in working or not matches(working[name],data,name):
                raise ProtocolError(f'committed immutable source was changed/deleted: {name}')

    def _index(self):
        path = self.path('index')
        if path.is_symlink():
            raise ProtocolError('Git index must not be a symlink')
        return path

    def index_bytes(self):
        path = self._index()
        return path.read_bytes() if path.exists() else None

    def status(self):
        raw = git_process(self.root,'status','--porcelain=v1','-z','--untracked-files=all','--ignore-submodules=none')
        if raw.returncode:
            raise ProtocolError('cannot inspect Git checkout status')
        entries = raw.stdout.split(b'\0'); result = {}; i = 0
        while i < len(entries) and entries[i]:
            item = entries[i];i += 1
            code,name = item[:2].decode(),os.fsdecode(item[3:])
            result[name] = code
            if 'R' in code or 'C' in code:
                i += 1  # porcelain -z gives the old name in a second field
        return result

    def _operation_check(self, *, own_index_lock=False):
        operations = ('MERGE_HEAD','CHERRY_PICK_HEAD','REVERT_HEAD','rebase-merge','rebase-apply','sequencer')
        active = [name for name in operations if self.path(name).exists()]
        if active:
            raise ProtocolError('unfinished Git operation ('+', '.join(active)+'); finish or abort it yourself before publication')
        if not own_index_lock and self.path('index.lock').exists():
            raise ProtocolError('Git index.lock exists; wait for the Git operation or inspect a stale lock before retrying')
        if git(self.root,'ls-files','--unmerged'):
            raise ProtocolError('unresolved Git conflicts; resolve and commit them before publication')

    def config_bytes(self):
        proc = git_process(self.root,'config','--null','--list','--show-origin')
        if proc.returncode:
            raise ProtocolError('cannot inspect Git configuration')
        return proc.stdout

    def upstream(self, branch):
        ref = git(self.root,'for-each-ref','--format=%(upstream)',branch)
        if not ref:
            return {'ref':None,'local_ahead':None,'upstream_ahead':None,
                    'notice':'No upstream configured; git push may require the user to select a remote/branch.'}
        known = git_process(self.root,'rev-parse','--verify',ref)
        if known.returncode:
            return {'ref':ref,'local_ahead':None,'upstream_ahead':None,
                    'notice':'Configured upstream is not available locally; freshness was not checked over the network.'}
        ahead,behind = map(int,git(self.root,'rev-list','--left-right','--count',f'HEAD...{ref}').split())
        if behind:
            reason = 'has diverged from' if ahead else 'is behind'
            raise ProtocolError(f'local branch {reason} its locally known upstream {ref} ({behind} upstream commits); update the checkout yourself before publication. No fetch, pull or rebase was performed')
        return {'ref':ref,'oid':known.stdout.decode().strip(),'local_ahead':ahead,'upstream_ahead':behind,
                'notice':'Only local upstream information was checked; no network freshness check was made.'}

    def preflight(self):
        self._operation_check()
        status = self.status()
        if status:
            detail = ', '.join(f'{code} {name!r}' for name,code in sorted(status.items()))
            raise ProtocolError('--commit requires a clean checkout (staged, unstaged and untracked nonignored changes found): '+detail+
                                '. Commit, stash or move these files yourself; without --commit publication remains available for manual Git review')
        if git(self.root,'ls-files','--','preview','site-build'):
            raise ProtocolError('site-build/ or preview/ contains tracked generated output; remove it from the index and commit that correction yourself before publication')
        for name in ('site-build/','preview/','.perseval-results.lock','.perseval-preview-preflight/','.perseval-site-preflight/'):
            if git_process(self.root,'check-ignore','--quiet','--no-index','--',name).returncode:
                raise ProtocolError(f'{name} must be ignored by Git; restore the repository .gitignore rules and commit that setup correction before publication')
        head = self.head()
        if head is None:
            raise ProtocolError('--commit requires the results repository sources to have an initial commit; commit setup separately first')
        branch_proc = git_process(self.root,'symbolic-ref','-q','HEAD')
        if branch_proc.returncode:
            raise ProtocolError('--commit requires a branch checkout; select the intended branch yourself before publishing')
        branch = branch_proc.stdout.decode().strip()
        for role in ('AUTHOR','COMMITTER'):
            proc = git_process(self.root,'var',f'GIT_{role}_IDENT')
            configured = []
            for key in ('name','email'):
                value = os.environ.get(f'GIT_{role}_{key.upper()}')
                if value is None:
                    value = git_process(self.root,'config','--get',f'{role.lower()}.{key}').stdout.decode().strip()
                    value = value or git_process(self.root,'config','--get',f'user.{key}').stdout.decode().strip()
                configured.append(value)
            if not all(value and value.strip() for value in configured) or proc.returncode or not re.search(rb'[^<>\n]+ <[^<>\s]+>',proc.stdout):
                raise ProtocolError(f'Git {role.lower()} identity is missing/invalid; configure git config user.name "Your name" and git config user.email "you@example.org" before publication')
        hidden = git(self.root,'ls-files','-v').splitlines()
        if any(line[:1]=='S' or line[:1].islower() for line in hidden):
            raise ProtocolError('--commit cannot verify a sparse/skip-worktree or assume-unchanged checkout; use a full checkout and clear these flags first')
        return GitState(head,branch,self.index_bytes(),self.config_bytes(),self.upstream(branch))

    def assert_state(self, state, allowed=None, *, own_index_lock=False):
        self._operation_check(own_index_lock=own_index_lock)
        branch = git_process(self.root,'symbolic-ref','-q','HEAD').stdout.decode().strip()
        if (self.head()!=state.head or branch!=state.branch or self.index_bytes()!=state.index
                or self.config_bytes()!=state.config or self.upstream(branch)!=state.upstream):
            raise ProtocolError('Git state changed concurrently after preflight; stop other Git operations and review the checkout before retrying')
        allowed = allowed or {}
        changed = self.status()
        if any(name not in allowed or code not in ('??',' M') for name,code in changed.items()):
            raise ProtocolError('checkout changed concurrently after preflight; preserve/review those changes before retrying')
        for name,data in allowed.items():
            path = self.root/name
            if path.is_symlink() or not path.is_file() or path.read_bytes()!=data:
                raise ProtocolError(f'planned publication file changed concurrently: {name}; inspect it before retrying')

    def filters(self, names):
        names = sorted(names)
        if not names:
            return {}
        proc = git_process(self.root,'check-attr','-z','filter','--',*names)
        if proc.returncode:
            raise ProtocolError('cannot inspect Git file attributes')
        fields = proc.stdout.split(b'\0')
        return {os.fsdecode(fields[i]):fields[i+2].decode() for i in range(0,len(fields)-1,3)}

    def plan_lfs(self, files):
        filters = self.filters(files)
        required = sorted(name for name,data in files.items() if eligible(name) and len(data)>=LFS_THRESHOLD)
        lfs = sorted(set(required) | {name for name,value in filters.items() if value=='lfs'})
        if any(not eligible(name) for name in lfs):
            raise ProtocolError('Git LFS must apply only to predictions/explicit artifacts; keep metadata, schemas and registries in ordinary Git. Correct .gitattributes first')
        attrs = self.root/'.gitattributes'
        if attrs.is_symlink():
            raise ProtocolError('.gitattributes must not be a symlink')
        additions = [name for name in required if filters.get(name)!='lfs']
        attribute_write = {}
        if additions:
            old = attrs.read_bytes() if attrs.exists() else b''
            suffix = (b'\n' if old and not old.endswith(b'\n') else b'')
            suffix += b'# Exact publication payload paths; metadata remains ordinary Git.\n'
            suffix += ''.join(f'/{name} filter=lfs diff=lfs merge=lfs -text\n' for name in additions).encode()
            attribute_write['.gitattributes'] = old+suffix
        if lfs:
            proc = git_process(self.root,'lfs','version')
            config = {key:git_process(self.root,'config','--get',f'filter.lfs.{key}').stdout.decode().strip()
                      for key in ('clean','process','required')}
            if proc.returncode or not config['clean'] or not config['process'] or config['required'].lower()!='true':
                raise ProtocolError('Git LFS is required for planned prediction/artifact files. Install Git LFS and run git lfs install --local yourself, then retry; no files were published')
            hook = self.path('hooks/pre-push')
            if (not hook.is_file() or not os.access(hook,os.X_OK)
                    or not re.search(rb'\bgit(?:\s+lfs|-lfs)\s+pre-push\b',hook.read_bytes())):
                raise ProtocolError('Git LFS requires an executable pre-push hook delegating to git lfs pre-push. Run git lfs install --local yourself (or integrate that delegation into your custom hooks), then retry; no files were published')
        return attribute_write,lfs

    def commit(self, files: dict[str,bytes], message: str, state: GitState):
        """Hold the real index lock; stage into a private index so user staging is untouched."""
        self.assert_state(state,files)
        index = self._index();lock = index.with_name(index.name+'.lock')
        try:
            lock_fd = os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        except FileExistsError as exc:
            raise ProtocolError('Git index changed concurrently (index.lock exists); retry after reviewing the checkout') from exc
        lock_identity = os.fstat(lock_fd)
        def release_lock():
            try: current = lock.lstat()
            except FileNotFoundError: return
            if (current.st_dev,current.st_ino)==(lock_identity.st_dev,lock_identity.st_ino):
                lock.unlink()
        before = state.head;proc = None;private = None;attempted = False;durable = None
        try:
            fd,name = tempfile.mkstemp(prefix='perseval-index-',dir=index.parent)
            private = Path(name)
            with os.fdopen(fd,'wb') as stream:
                stream.write(state.index)
            staged = git_process(self.root,'add','--',*sorted(files),private_index=private)
            if staged.returncode:
                raise ProtocolError('git add failed: '+staged.stderr.decode(errors='replace').strip())
            diff = git_process(self.root,'diff','--cached','--name-only','-z',private_index=private)
            names = {os.fsdecode(n) for n in diff.stdout.split(b'\0') if n}
            if diff.returncode or names!=set(files):
                raise ProtocolError('staged diff does not match the exact planned file allowlist')
            modes = git_process(self.root,'ls-files','--stage','-z','--',*sorted(files),private_index=private)
            if modes.returncode or any(item.split(b' ',1)[0]!=b'100644'
                                       for item in modes.stdout.split(b'\0') if item):
                raise ProtocolError('planned publication files must be regular non-executable Git blobs')
            filters = self.filters(files)
            if any(eligible(name) and len(data)>=LFS_THRESHOLD and filters.get(name)!='lfs'
                   for name,data in files.items()):
                raise ProtocolError('required Git LFS rules were overridden; inspect .git/info/attributes and nested .gitattributes before retrying')
            expected = {name:pointer_bytes(data) if filters.get(name)=='lfs' else data for name,data in files.items()}
            for name,data in expected.items():
                staged = git_process(self.root,'show',f':{name}',private_index=private)
                if staged.returncode or staged.stdout!=data:
                    raise ProtocolError(f'Git staged bytes differ from validated source (check filters): {name}')
            self.assert_state(state,files,own_index_lock=True)
            attempted = True
            proc = git_process(self.root,'commit','-m',message,private_index=private)
            after = self.head()
            if proc.returncode == 0 or after!=before:
                sha = after or 'unavailable (inspect git log)';durable = sha;warnings = []
                if proc.returncode:
                    warnings.append('Git reported failure after HEAD advanced; the new commit is durable and was not rolled back.')
                # HEAD is durable before updating the real index or inspecting the commit.
                if self.index_bytes()==state.index:
                    with os.fdopen(os.dup(lock_fd),'wb') as stream:
                        stream.write(private.read_bytes());stream.flush();os.fsync(stream.fileno())
                    os.replace(lock,index)
                else:
                    warnings.append('Real Git index changed despite index.lock; left user staging untouched. Inspect git status before pushing.')
                committed = self.head_files(after)
                for name in files:
                    if name=='.gitattributes':
                        committed[name] = git_process(self.root,'show',f'{after}:{name}').stdout
                if any(committed.get(name)!=data for name,data in expected.items()):
                    warnings.append('Committed managed bytes differ from the validated plan (possibly a Git hook). Inspect the durable commit before pushing.')
                actual = git(self.root,'diff-tree','--no-commit-id','--name-only','-r',after).splitlines()
                if set(actual)!=set(files):
                    warnings.append('Durable commit paths differ from the planned allowlist (possibly a Git hook); inspect before pushing.')
                if self.status():
                    warnings.append('Commit is durable, but the checkout changed during commit (possibly a Git hook or editor); review git status before pushing.')
                return CommitOutcome(sha,warnings)
            raise ProtocolError('git commit failed: '+proc.stderr.decode(errors='replace').strip())
        except BaseException as exc:
            try: after = self.head()
            except Exception: after = before
            if (attempted and proc is not None and proc.returncode==0) or after!=before:
                durable = after or 'unavailable (inspect git log)'
                raise DurableCommitError(durable,str(exc)) from exc
            raise
        finally:
            failures = []
            for cleanup in (release_lock,lambda:os.close(lock_fd),
                            lambda:private.unlink(missing_ok=True) if private else None,
                            lambda:private.with_name(private.name+'.lock').unlink(missing_ok=True) if private else None):
                try: cleanup()
                except OSError as exc: failures.append(str(exc))
            if failures:
                message = 'Git transaction cleanup failed: '+'; '.join(failures)
                if durable:
                    raise DurableCommitError(durable,message)
                raise ProtocolError(message)
