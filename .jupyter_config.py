# Password to use for web authentication
# the password can be generated with:
# >>>from notebook.auth import passwd
# >>>passwd("glotaran")
# c.NotebookApp.password = u'sha1:deef177aa2dd:657a28dd6c1a12eaf67aa6ab4dfb64166a6ba787'

c.NotebookApp.token = ''

c.NotebookApp.notebook_dir = '/vagrant/tests/notebooks'


# Set ip to '*' to bind on all interfaces (ips) for the public server
c.NotebookApp.ip = '*'

# Don't open browser since vagrant can't read it
c.NotebookApp.open_browser = False

# It is a good idea to set a known, fixed port for server access
c.NotebookApp.port = 9999