from setuptools import setup, find_packages

if __name__ == '__main__':
    setup(
        name='jpx1',
        description='python project for jpx1 competition',
        author='Hiroki Taniai',
        author_email='charmer.popopo@gmail.com',
        packages=find_packages(include=['src']),
    )
